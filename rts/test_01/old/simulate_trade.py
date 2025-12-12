#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединённый скрипт backtest + симулятор без промежуточных XLSX файлов.
Вместо сохранения файлов используется трехмерный numpy-массив.

Структура 3D массива:
    A[day_index, row_index, model_index]
        day_index   — номер обработанного дня (0..N-1)
        row_index   — строка внутри результата backtest (кол-во строк в df)
        model_index — индекс модели (0..27) соответствующий max_3..max_30

Выход:
    • result.xlsx — итоговая симуляция
    • plot_prev_max_cumulative.png — график prev_max и P/L

"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import hashlib
import sqlite3
import yaml
import logging
from langchain_core.documents import Document
import matplotlib.pyplot as plt

# ==========================================================
#                       ЗАГРУЗКА НАСТРОЕК
# ==========================================================
SETTINGS_FILE = Path(__file__).parent.parent / "settings.yaml"
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings.get('ticker', "RTS")
ticker_lc = ticker.lower()
provider = settings.get('provider', 'investing')
min_prev_files = settings.get('min_prev_files', 2)
test_days = settings.get('test_days', 22) + 1
md_path = Path(settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))

# ==========================================================
#                       ЛОГИРОВАНИЕ
# ==========================================================
logger = logging.getLogger("combined")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ==========================================================
#                 ФУНКЦИИ ИЗ multi_xlsx_01.py
# ==========================================================

def cosine_similarity(vec1, vec2):
    """Косинусное сходство между двумя векторами."""
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0


def load_markdown_files(directory):
    """Загрузка Markdown-файлов и разбор метаданных."""
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
    docs = []
    for file_path in files:
        content = file_path.read_text(encoding='utf-8')
        md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1]
                text_content = parts[2]
                metadata = yaml.safe_load(metadata_yaml) or {}
                metadata = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),
                    "date": file_path.stem,
                    "source": file_path.name,
                    "md5": md5_hash
                }
                docs.append(Document(page_content=text_content, metadata=metadata))
    return docs


def load_quotes(path_db_quote):
    """Загрузка котировок и расчёт next_bar_pips."""
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)

    df = df.sort_values('TRADEDATE')
    df['next_bar_pips'] = (df['CLOSE'] - df['OPEN']).shift(-1)
    df = df.dropna(subset=['next_bar_pips'])
    return df.set_index('TRADEDATE')[['next_bar_pips']]


def load_cache(cache_file):
    """Загрузка кэша эмбеддингов."""
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def backtest_for_docs(documents, cache, quotes_df, max_prev_files):
    """Возвращает df вида: [test_date, cumulative] для конкретного max_prev_files."""
    rows = []

    for test_doc in documents[min_prev_files:]:
        real_next_bar = test_doc.metadata['next_bar']
        test_date = test_doc.metadata['date']
        test_id = test_doc.metadata['md5']

        if real_next_bar in ('unknown', 'None'):
            continue

        # Ищем embedding текущего файла
        test_emb = next((x['embedding'] for x in cache if x['id'] == test_id), None)
        if test_emb is None:
            continue

        # Находим предыдущие документы
        prev_cache = sorted(
            [x for x in cache if x['metadata']['date'] < test_date],
            key=lambda x: x['metadata']['date'],
            reverse=True
        )[:max_prev_files]

        if len(prev_cache) < min_prev_files:
            continue

        # Косинусные сходства
        sims = [(cosine_similarity(test_emb, x['embedding']), x['metadata']) for x in prev_cache]
        sims.sort(key=lambda x: x[0], reverse=True)

        predicted_next_bar = sims[0][1]['next_bar']
        is_correct = predicted_next_bar == real_next_bar

        try:
            p = quotes_df.loc[test_date, 'next_bar_pips']
        except KeyError:
            continue

        next_bar_pips = abs(p) if is_correct else -abs(p)
        rows.append({"test_date": test_date, "value": next_bar_pips})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df["cumulative"] = df["value"].cumsum()
    return df

# ==========================================================
#                   НОВЫЙ ОБЪЕДИНЁННЫЙ КОНВЕЙЕР
# ==========================================================

def run_backtest_to_numpy():
    """
    Проводит backtest по всем датам и формирует список 2D numpy массивов,
    соответствующих ежедневным датафреймам.

    Возвращает:
        dates: список дат
        all_daily_arrays: список 2D массивов (rows × models)
    """

    quotes_df = load_quotes(path_db_day)
    documents = load_markdown_files(md_path)
    cache = load_cache(cache_file)

    if len(documents) < 5:
        raise ValueError("Недостаточно markdown-файлов (<5)")

    dates = []
    daily_arrays = []

    for end_idx in range(4, len(documents)):
        date_str = documents[end_idx].metadata['date']
        logger.info(f"Обработка даты {date_str}")

        # Ограничение test_days
        cut_docs = documents[:end_idx + 1]
        start_idx = max(min_prev_files, len(cut_docs) - test_days)
        cut_docs_effective = cut_docs[start_idx:]

        # Собираем все модели max_3..max_30
        cols = []
        for max_p in range(3, 31):
            df = backtest_for_docs(cut_docs_effective, cache, quotes_df, max_p)
            if df.empty:
                continue
            df = df.rename(columns={"cumulative": f"max_{max_p}"})
            cols.append(df[["test_date", f"max_{max_p}"]])

        if not cols:
            continue

        # Объединяем все модели по test_date
        merged = cols[0]
        for c in cols[1:]:
            merged = merged.merge(c, on="test_date", how="outer")

        merged = merged.sort_values("test_date").reset_index(drop=True)

        # Конвертация в numpy (только столбцы моделей)
        model_cols = [f"max_{i}" for i in range(3, 31) if f"max_{i}" in merged.columns]
        arr = merged[model_cols].to_numpy(dtype=float)

        dates.append(date_str)
        daily_arrays.append(arr)

    return dates, daily_arrays


# ==========================================================
#                 СИМУЛЯЦИОННАЯ ТОРГОВЛЯ
# ==========================================================

def run_simulation(dates, daily_arrays):
    """
    Логика симуляции как в processing_xlsx.py, но работает с numpy-массивами.

    Параметры:
        dates — список дат
        daily_arrays — список 2D numpy массивов (rows × models)

    Возвращает:
        df_rez — итоговый DataFrame с P/L
    """

    results = []

    for i in range(1, len(daily_arrays)):
        prev_arr = daily_arrays[i - 1]
        curr_arr = daily_arrays[i]
        day = dates[i]

        # Индекс строки последнего значения предыдущего дня
        prev_last = prev_arr[-1]

        # Находим лучшую модель на предыдущем дне
        prev_best_idx = np.nanargmax(prev_last)

        # Текущие значения
        last_val = curr_arr[-1, prev_best_idx]
        second_last_val = curr_arr[-2, prev_best_idx]
        diff = last_val - second_last_val

        model_number = prev_best_idx + 3  # т.к. модели идут max_3..max_30

        results.append({
            "Дата": day,
            "prev_max": model_number,
            "Profit/Loss": diff
        })

    df_rez = pd.DataFrame(results)
    df_rez["Cumulative Profit/Loss"] = df_rez["Profit/Loss"].cumsum()
    return df_rez

# ==========================================================
#                       ПОСТРОЕНИЕ ГРАФИКА
# ==========================================================

def plot_results(df_rez, out_path):
    """Строит график prev_max и cumulative P/L."""
    df_rez['Дата'] = pd.to_datetime(df_rez['Дата'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(df_rez['Дата'], df_rez['prev_max'], color='blue', label='prev_max', width=0.6)

    ax2 = ax1.twinx()
    ax2.plot(df_rez['Дата'], df_rez['Cumulative Profit/Loss'], color='green', marker='o',
             linewidth=2, label='Cumulative Profit/Loss')

    ax1.set_xlabel('Дата')
    ax1.set_ylabel('prev_max', color='blue')
    ax2.set_ylabel('Cumulative Profit/Loss', color='green')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'{ticker} prev_max и Cumulative Profit/Loss')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================================
#                       ГЛАВНАЯ ФУНКЦИЯ
# ==========================================================

def main():
    logger.info("=== Запуск объединённого конвейера ===")

    dates, daily_arrays = run_backtest_to_numpy()
    logger.info(f"Получено дней: {len(dates)}")

    df_rez = run_simulation(dates, daily_arrays)

    out_result = Path(__file__).parent / "result.xlsx"
    df_rez.to_excel(out_result, index=False)
    logger.info(f"Итог сохранён в {out_result}")

    out_plot = Path(__file__).parent / "plot_prev_max_cumulative.png"
    plot_results(df_rez, out_plot)
    logger.info(f"График сохранён в {out_plot}")


if __name__ == "__main__":
    main()
