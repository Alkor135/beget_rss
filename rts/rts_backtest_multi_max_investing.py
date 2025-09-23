"""
Скрипт для проведения backtests с разными значениями max_prev_files от 4 до 30.
Сохраняет только даты и cumulative_next_bar_pips для каждого значения в один XLSX файл на один лист.
Колонки: test_date, max_4, max_5, ..., max_30.

⚡️Изменено:
- Убрана логика пересчёта и обновления pkl (кэш должен существовать заранее).
- Теперь кэш только читается. Если pkl нет → ошибка.
- Учитывается, что в кэше id = md5(page_content).
"""

import pandas as pd
from pathlib import Path
import pickle
import hashlib
import numpy as np
import sqlite3
from langchain_core.documents import Document
import logging
import yaml

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
ticker_lc = ticker.lower()
provider = settings['provider']  # Провайдер RSS новостей
min_prev_files = settings['min_prev_files']  # Минимальное количество предыдущих файлов

md_path = Path(  # Путь к markdown-файлам
    settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(
    settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
output_dir = Path(  # Путь к папке с результатами
    settings['output_dir'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
log_file = Path(  # Путь к файлу лога
    output_dir / 'log' / # Папка для логов
    fr'{ticker_lc}_backtest_multi_max_{provider}.txt')
output_file = Path(  # Итоговый XLSX файл
    fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}'
    fr'\{ticker_lc}_backtest_results_multi_max_{provider}.xlsx')

# Настройка логирования
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def cosine_similarity(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def load_markdown_files(directory):
    """Загружает все MD-файлы из директории, сортирует по дате (имени файла)."""
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
    documents = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # # ⚡ Добавляем md5 от всего содержимого файла (YAML + текст)
        md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()  # нового md5 разкомментить

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1].strip()
                text_content = parts[2].strip()
                # md5_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()  # нового md5 закомментить
                metadata = yaml.safe_load(metadata_yaml) or {}
                metadata_str = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),
                    "date_min": str(metadata.get("date_min", "unknown")),
                    "date_max": str(metadata.get("date_max", "unknown")),
                    "source": file_path.name,
                    "date": file_path.stem,
                    "md5": md5_hash  # ⚡ сохраняем md5 в метаданные
                }
                doc = Document(page_content=text_content, metadata=metadata_str)
                documents.append(doc)
    return documents

def load_quotes(path_db_quote):
    """Читает таблицу Futures из базы данных котировок и возвращает DataFrame с next_bar_pips."""
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)
    df = df.sort_values('TRADEDATE', ascending=True)
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['next_bar_pips'] = df.apply(lambda x: (x['CLOSE'] - x['OPEN']), axis=1).shift(-1)
    df = df.dropna(subset=['next_bar_pips'])
    return df[['TRADEDATE', 'next_bar_pips']].set_index('TRADEDATE')


def load_cache(cache_file):
    """
    Загружает готовый кэш эмбеддингов.
    ⚡️Изменено: здесь больше нет пересчёта, только чтение.
    """
    if not cache_file.exists():
        logger.error(f"Ошибка: кэш {cache_file} не найден. Сначала сгенерируй его другим скриптом.")
        exit(1)

    logger.info(f"Загрузка кэша эмбеддингов из {cache_file}")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    return cache


def backtest_predictions(documents, cache, quotes_df, max_prev_files):
    """Проводит backtesting для заданного max_prev_files и возвращает DataFrame с test_date и cumulative_next_bar_pips."""
    results = []

    for test_idx in range(min_prev_files, len(documents)):
        test_doc = documents[test_idx]
        real_next_bar = test_doc.metadata['next_bar']
        test_date = test_doc.metadata['date']

        if real_next_bar == 'unknown' or real_next_bar == 'None':
            continue

        # Получение эмбеддинга тестовой даты из кэша (по md5)
        test_id = test_doc.metadata.get("md5")  # ⚡ теперь берём md5 из метаданных
        test_embedding = next((item['embedding'] for item in cache if item['id'] == test_id), None)
        if test_embedding is None:
            continue

        # Предыдущие документы из кэша
        prev_cache = sorted(
            [item for item in cache if item['metadata']['date'] < test_date],
            key=lambda x: x['metadata']['date'], reverse=True
        )[:max_prev_files]

        if len(prev_cache) < min_prev_files:
            continue

        # Вычисление сходств
        similarities = [
            (cosine_similarity(test_embedding, item['embedding']) * 100, item['metadata'])
            for item in prev_cache
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Ближайший документ
        if similarities:
            closest_similarity, closest_metadata = similarities[0]
            predicted_next_bar = closest_metadata['next_bar']
            is_correct = predicted_next_bar == real_next_bar

            try:
                next_bar_pips_value = quotes_df.loc[test_date, 'next_bar_pips']
                next_bar_pips = abs(next_bar_pips_value) if is_correct else -abs(next_bar_pips_value)
            except KeyError:
                continue

            results.append({
                'test_date': test_date,
                'next_bar_pips': next_bar_pips
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['cumulative_next_bar_pips'] = results_df['next_bar_pips'].cumsum()
        return results_df[['test_date', 'cumulative_next_bar_pips']]
    else:
        return pd.DataFrame()


def main():
    # Загрузка котировок
    if not path_db_day.exists():
        logger.error(f"Ошибка: Файл базы данных котировок не найден: {path_db_day}")
        exit(1)
    quotes_df = load_quotes(path_db_day)

    # Загрузка markdown-файлов
    documents = load_markdown_files(md_path)
    if len(documents) < min_prev_files + 1:
        logger.error(f"Недостаточно файлов: {len(documents)}. Требуется минимум {min_prev_files + 1}.")
        exit(1)

    # ⚡️Загрузка готового кэша (без пересчёта)
    cache = load_cache(cache_file)

    # Создание итогового DataFrame
    all_results = pd.DataFrame()

    for max_prev in range(3, 31):  # от 4 до 30
        logger.info(f"Проводим backtest для max_prev_files = {max_prev}")
        results_df = backtest_predictions(documents, cache, quotes_df, max_prev)
        if not results_df.empty:
            results_df = results_df.rename(columns={'cumulative_next_bar_pips': f'max_{max_prev}'})
            if all_results.empty:
                all_results = results_df
            else:
                all_results = all_results.merge(results_df, on='test_date', how='outer')

    if not all_results.empty:
        all_results.to_excel(output_file, index=False, engine='openpyxl')
        logger.info(f"Результаты сохранены в {output_file}")
    else:
        logger.error("Нет результатов для сохранения.")

if __name__ == '__main__':
    main()