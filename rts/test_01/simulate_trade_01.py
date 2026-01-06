#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_trade_combined.py

Полный объединённый скрипт:
- выполняет backtest (логика из multi_xlsx_01.py)
- формирует в памяти набор ежедневных 2D numpy-массивов (rows x models)
- запускает симуляцию (логика из processing_xlsx.py), но использует массивы вместо xlsx
- сохраняет только итог (result.xlsx) и график

Изменения/особенности:
- Включена переменная START_DATE (как в processing_xlsx.py) — берётся из settings.yaml или по умолчанию.
- Комментарии:
    * секции, которые я не менял — помечены "# (НЕ МЕНЯЛОСЬ)"
    * в местах изменений — подробные комментарии на русском
"""

from pathlib import Path
from datetime import datetime
import pickle
import hashlib
import sqlite3
import logging
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# langchain_core.documents.Document используется в оригинале для хранения md-файлов
# Если у тебя нет этой зависимости — можно заменить простой dict-структурой.
try:
    from langchain_core.documents import Document
except Exception:
    # Если нет langchain — используем простой объект-подстановку
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

# =======================
# ЗАГРУЗКА НАСТРОЕК (НЕ МЕНЯЛОСЬ, только добавлена START_DATE)
# =======================
BASE = Path(__file__).resolve().parent
SETTINGS_FILE = BASE.parent / "settings.yaml"

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

# --------- START_DATE (как в processing_xlsx.py) ----------
# Значение читается из settings.yaml как 'start_date' в формате YYYY-MM-DD,
# если не указано — используется значение, которое было в оригинальном processing_xlsx.py.
START_DATE = settings.get('start_date', "2025-07-30")
START_DT = datetime.strptime(START_DATE, "%Y-%m-%d").date()
# -----------------------------------------------------------------

# =======================
# ЛОГИРОВАНИЕ
# =======================
logger = logging.getLogger("simulate_combined")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# ========================================
# ФУНКЦИИ: косинус/загрузка md/quotes/cache
# ========================================
def cosine_similarity(vec1, vec2):
    """Косинусное сходство."""
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

def load_markdown_files(directory):
    """Загрузка markdown-файлов в объект Document."""
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
    documents = []
    for file_path in files:
        content = file_path.read_text(encoding='utf-8')
        md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1]
                body = parts[2]
                metadata_parsed = yaml.safe_load(metadata_yaml) or {}
                metadata = {
                    "next_bar": str(metadata_parsed.get("next_bar", "unknown")),
                    "date": file_path.stem,
                    "source": file_path.name,
                    "md5": md5_hash
                }
                documents.append(Document(page_content=body, metadata=metadata))
    return documents

def load_quotes(path_db_quote):
    """Загрузка котировок и расчет next_bar_pips."""
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)
    df = df.sort_values('TRADEDATE')
    df['next_bar_pips'] = (df['CLOSE'] - df['OPEN']).shift(-1)
    df = df.dropna(subset=['next_bar_pips'])
    # индексируем по TRADEDATE (строки типа YYYY-MM-DD)
    return df.set_index('TRADEDATE')[['next_bar_pips']]

def load_cache(cache_file_path):
    """Загрузка кэша эмбеддингов."""
    with open(cache_file_path, 'rb') as f:
        return pickle.load(f)

# ========================================
# backtest_for_docs — оставляем логику
# ========================================
def backtest_for_docs(documents, cache, quotes_df, max_prev_files):
    """
    Выполняет backtest для набора документов (входных cut_docs_effective)
    и возвращает DataFrame с колонками [test_date, cumulative] для данного max_prev_files.
    """
    rows = []
    for test_doc in documents[min_prev_files:]:
        real_next_bar = test_doc.metadata['next_bar']
        test_date = test_doc.metadata['date']
        test_id = test_doc.metadata['md5']
        if real_next_bar in ('unknown', 'None'):
            continue
        test_emb = next((x['embedding'] for x in cache if x['id'] == test_id), None)
        if test_emb is None:
            continue
        prev_cache = sorted(
            [x for x in cache if x['metadata']['date'] < test_date],
            key=lambda x: x['metadata']['date'],
            reverse=True
        )[:max_prev_files]
        if len(prev_cache) < min_prev_files:
            continue
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

# ========================================
# НОВАЯ: run_backtest_to_numpy (упрощено и добавлена фильтрация START_DATE)
# ========================================
def run_backtest_to_numpy():
    """
    Выполняет проход по датам (как в original multi_xlsx_01), но:
    - НЕ сохраняет XLSX
    - Формирует список ежедневных 2D numpy массивов (rows x models).
    - ПРОПУСКАЕТ даты до START_DATE (включительно START_DATE будет обработан).
    Возвращает: (dates_list, daily_arrays_list)
    """
    # загрузка данных
    quotes_df = load_quotes(path_db_day)
    documents = load_markdown_files(md_path)
    cache = load_cache(cache_file)

    if len(documents) < 5:
        raise RuntimeError("Недостаточно markdown файлов (<5)")

    dates = []
    daily_arrays = []

    # Перебираем end_idx аналогично оригиналу
    for end_idx in range(4, len(documents)):
        date_str = documents[end_idx].metadata['date']
        date_dt = datetime.strptime(date_str, "%Y-%m-%d").date()

        # ---- фильтрация по START_DATE ----
        # Пропускаем обработку дат раньше START_DATE.
        if date_dt < START_DT:
            continue
        # -----------------------------------------

        logger.info(f"Backtest: обрабатываем дату {date_str}")

        cut_docs = documents[:end_idx + 1]
        # логика test_days
        start_idx = max(min_prev_files, len(cut_docs) - test_days)
        cut_docs_effective = cut_docs[start_idx:]

        # Собираем результаты по каждой модели max_3..max_30 (каждый — DataFrame со столбцами test_date и max_N)
        model_dfs = []
        for max_p in range(3, 31):
            df = backtest_for_docs(cut_docs_effective, cache, quotes_df, max_p)
            if df.empty:
                continue
            df = df.rename(columns={"cumulative": f"max_{max_p}"})
            model_dfs.append(df[["test_date", f"max_{max_p}"]])

        if not model_dfs:
            logger.warning(f"Нет данных для даты {date_str} (все модели вернули пустые результаты).")
            continue

        # Объединяем все model_dfs по test_date (outer merge)
        merged = model_dfs[0]
        for df_m in model_dfs[1:]:
            merged = merged.merge(df_m, on="test_date", how="outer")

        merged = merged.sort_values("test_date").reset_index(drop=True)

        # Приведение названий колонок к ожидаемому порядку max_3..max_30
        model_cols = [f"max_{i}" for i in range(3, 31) if f"max_{i}" in merged.columns]
        arr = merged[model_cols].to_numpy(dtype=float)  # rows x models

        # Добавляем только если достаточно строк (симулятор предполагает минимум 2 строки для расчетов)
        if arr.shape[0] < 2:
            logger.warning(f"Для даты {date_str} недостаточно строк ({arr.shape[0]}) для симуляции. Пропускаем.")
            continue

        dates.append(date_str)
        daily_arrays.append(arr)

    return dates, daily_arrays

# ========================================
# СИМУЛЯЦИЯ (упрощена, но логика совпадает с original processing_xlsx.py)
# ========================================
def run_simulation(dates, daily_arrays):
    """
    Для каждой даты начиная со второй:
    - берем prev_arr = daily_arrays[i-1], curr_arr = daily_arrays[i]
    - определяем prev_best_idx = индекс максимальной модели в последней строке prev_arr
    - вычисляем diff = curr_arr[-1, prev_best_idx] - curr_arr[-2, prev_best_idx]
    Формируем DataFrame с колонками: Дата, prev_max, Profit/Loss, Cumulative Profit/Loss
    """
    results = []

    # Если нет данных или только один день — нечего симулировать
    if len(daily_arrays) < 2:
        logger.warning("Недостаточно дней для симуляции (меньше 2). Возвращаем пустой результат.")
        return pd.DataFrame(columns=["Дата", "prev_max", "Profit/Loss", "Cumulative Profit/Loss"])

    for i in range(1, len(daily_arrays)):
        prev_arr = daily_arrays[i - 1]
        curr_arr = daily_arrays[i]
        day = dates[i]

        # Защита: убедимся, что предыдущий и текущий массивы имеют >=2 строк и одинаковое число моделей
        if prev_arr.shape[0] < 1 or curr_arr.shape[0] < 2 or prev_arr.shape[1] != curr_arr.shape[1]:
            logger.warning(
                f"Пропускаем день {day}: некорректные размеры prev({prev_arr.shape})/"
                f"curr({curr_arr.shape})"
            )
            continue

        # Если в prev_arr последняя строка все NaN — nanargmax упадёт; обработаем этот кейс:
        prev_last = prev_arr[-1]
        if np.all(np.isnan(prev_last)):
            logger.warning(
                f"Все значения последней строки предыдущего дня NaN для дня {day}. Пропускаем."
            )
            continue

        # Индекс лучшей модели (если ties — берётся первый max)
        prev_best_idx = int(np.nanargmax(prev_last))

        # Берём последние два значения выбранной модели в текущем дне
        last_val = curr_arr[-1, prev_best_idx]
        second_last_val = curr_arr[-2, prev_best_idx]

        # Если значения NaN — пропускаем
        if np.isnan(last_val) or np.isnan(second_last_val):
            logger.warning(
                f"NaN значения для выбранной модели на дате {day} (model idx {prev_best_idx}). "
                f"Пропускаем."
            )
            continue

        diff = float(last_val - second_last_val)
        model_number = prev_best_idx + 3  # т.к. модели индексируются от max_3

        results.append({
            "Дата": day,
            "prev_max": int(model_number),
            "Profit/Loss": diff
        })

    df_rez = pd.DataFrame(results)
    if not df_rez.empty:
        df_rez["Cumulative Profit/Loss"] = df_rez["Profit/Loss"].cumsum()
    else:
        # Пустой DF — создаём нужные колонки
        df_rez = pd.DataFrame(columns=["Дата", "prev_max", "Profit/Loss", "Cumulative Profit/Loss"])
    return df_rez

# ========================================
# ПОСТРОЕНИЕ ГРАФИКА
# ========================================
def plot_results(df_rez, out_path):
    if df_rez.empty:
        logger.warning("Нет данных для графика — пропускаю построение.")
        return
    df_rez['Дата'] = pd.to_datetime(df_rez['Дата'])
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(df_rez['Дата'], df_rez['prev_max'], color='green', label='prev_max', width=0.6)
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('prev_max', color='green')

    ax2 = ax1.twinx()
    ax2.plot(df_rez['Дата'], df_rez['Cumulative Profit/Loss'], color='blue', marker='o',
             linewidth=2, label='Cumulative Profit/Loss')
    ax2.set_ylabel('Cumulative Profit/Loss', color='blue')
    ax2.axhline(0, linestyle='--', linewidth=1, color='black')  # Горизонтальная линия уровня 0

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'{ticker} prev_max и Cumulative Profit/Loss')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ========================================
# ГЛАВНАЯ: orchestration и сохранение результатов
# ========================================
def main():
    logger.info("=== Запуск combined backtest + simulation ===")
    logger.info(f"START_DATE = {START_DATE} (фильтрация ранних дат)")
    dates, daily_arrays = run_backtest_to_numpy()
    logger.info(f"Сгенерировано дней (после фильтрации): {len(dates)}")

    df_rez = run_simulation(dates, daily_arrays)
    out_result = BASE / "result.xlsx"
    df_rez.to_excel(out_result, index=False)
    logger.info(f"Итоговая таблица сохранена: {out_result}")

    current_date = datetime.now().strftime("%Y-%m")
    out_plot = BASE / f"{current_date}_prev_max_cumulative.png"
    plot_results(df_rez, out_plot)
    logger.info(f"График сохранён: {out_plot}")

if __name__ == "__main__":
    main()
