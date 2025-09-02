"""
Скрипт для проведения backtests с разными значениями max_prev_files от 10 до 30.
Сохраняет только даты и cumulative_next_bar_pips для каждого значения в один XLSX файл на один лист.
Колонки: test_date, max_10, max_11, ..., max_30.
"""

import pandas as pd
from pathlib import Path
import pickle
import hashlib
import numpy as np
import yaml
import sqlite3
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import logging

# Параметры
ticker: str = 'RTS'  # Тикер фьючерса
ticker_lc: str = 'rts'  # Тикер фьючерса в нижнем регистре
md_path = Path(fr'C:\Users\Alkor\gd\md_{ticker_lc}_investing')
cache_file = Path(fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\{ticker_lc}_embeddings_investing_ollama.pkl')
path_db_quote = Path(fr'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db')
model_name = "bge-m3"
url_ai = "http://localhost:11434/api/embeddings"
min_prev_files = 4   # Минимальное количество предыдущих файлов для предсказаний
# Итоговый XLSX файл
output_file = fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\{ticker_lc}_backtest_results_multi_max_investing.xlsx'

# Настройка логирования: вывод в консоль и в файл, файл перезаписывается
log_file = Path(
    fr'C:\Users\Alkor\gd\predict_ai\{ticker_lc}_investing_ollama\log\{ticker_lc}_backtest_multi_max_investing.txt')
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Удаляем существующие обработчики, чтобы избежать дублирования
logger.handlers = []
# logger = logging.getLogger('Predict.NextSessionInvesting')
logger.addHandler(logging.FileHandler(log_file))
# Обработчик для консоли
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
# Обработчик для файла (перезаписывается при каждом запуске)
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
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)  # Сортировка по дате ascending
    documents = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1].strip()
                text_content = parts[2].strip()
                metadata = yaml.safe_load(metadata_yaml) or {}
                metadata_str = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),
                    "date_min": str(metadata.get("date_min", "unknown")),
                    "date_max": str(metadata.get("date_max", "unknown")),
                    "source": file_path.name,
                    "date": file_path.stem
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
    # Вычисляем финансовый результат за следующий день
    df['next_bar_pips'] = df.apply(
        lambda x: (x['CLOSE'] - x['OPEN']), axis=1
    ).shift(-1)
    # Удаляем строки с NaN в next_bar_pips, если нужно
    df = df.dropna(subset=['next_bar_pips'])
    return df[['TRADEDATE', 'next_bar_pips']].set_index('TRADEDATE')

def cache_is_valid(documents, cache_file):
    """Проверяет, актуален ли кэш эмбеддингов."""
    if not cache_file.exists():
        return False

    cache_mtime = cache_file.stat().st_mtime
    current_files = {doc.metadata['source'] for doc in documents}

    # Загружаем кэш для проверки
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)

    cache_files = {item['metadata']['source'] for item in cache}

    # Проверяем, совпадают ли наборы файлов
    if current_files != cache_files:
        logger.info("Кэш устарел: изменился набор markdown-файлов.")
        return False

    # Проверяем, не изменились ли файлы
    for doc in documents:
        file_path = md_path / doc.metadata['source']
        if file_path.stat().st_mtime > cache_mtime:
            logger.info(f"Кэш устарел: файл {file_path.name} был изменён.")
            return False

    return True

def cache_embeddings(documents, cache_file, model_name, url_ai):
    """Вычисляет и кэширует эмбеддинги всех документов в pickle-файл."""
    if cache_is_valid(documents, cache_file):
        logger.info(f"Загрузка кэша эмбеддингов из {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        return cache

    logger.info("Вычисление эмбеддингов...")
    ef = OllamaEmbeddingFunction(model_name=model_name, url=url_ai)
    cache = []
    for doc in documents:
        embedding = ef([doc.page_content])[0]
        cache.append({
            'id': hashlib.md5(doc.page_content.encode()).hexdigest(),
            'embedding': embedding,
            'metadata': doc.metadata
        })
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    logger.info(f"Эмбеддинги сохранены в {cache_file}")
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

        # Получение эмбеддинга тестовой даты из кэша
        test_id = hashlib.md5(test_doc.page_content.encode()).hexdigest()
        test_embedding = None
        for item in cache:
            if item['id'] == test_id:
                test_embedding = item['embedding']
                break
        if test_embedding is None:
            continue

        # Получение предыдущих документов из кэша, ближайших по дате
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

        # Сортировка по убыванию сходства
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Ближайший документ
        if similarities:
            closest_similarity, closest_metadata = similarities[0]
            predicted_next_bar = closest_metadata['next_bar']
            is_correct = predicted_next_bar == real_next_bar

            # Получение next_bar_pips из базы котировок
            try:
                next_bar_pips_value = quotes_df.loc[test_date, 'next_bar_pips']
                next_bar_pips = abs(next_bar_pips_value) if is_correct else -abs(next_bar_pips_value)
            except KeyError:
                continue

            results.append({
                'test_date': test_date,
                'next_bar_pips': next_bar_pips
            })

    # Создание DataFrame и добавление накопительного результата
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['cumulative_next_bar_pips'] = results_df['next_bar_pips'].cumsum()
        return results_df[['test_date', 'cumulative_next_bar_pips']]
    else:
        return pd.DataFrame()

def main():
    # Загрузка котировок
    if not path_db_quote.exists():
        logger.error(f"Ошибка: Файл базы данных котировок не найден: {path_db_quote}")
        exit(1)
    quotes_df = load_quotes(path_db_quote)

    # Загрузка markdown-файлов
    documents = load_markdown_files(md_path)
    if len(documents) < min_prev_files + 1:
        logger.error(f"Недостаточно файлов: {len(documents)}. Требуется минимум {min_prev_files + 1}.")
        exit(1)

    # Кэширование эмбеддингов (один раз)
    cache = cache_embeddings(documents, cache_file, model_name, url_ai)

    # Создание итогового DataFrame с колонкой test_date
    all_results = pd.DataFrame()

    # Диапазон max_prev_files от 10 до 30
    for max_prev in range(5, 31):
        logger.info(f"Проводим backtest для max_prev_files = {max_prev}")
        results_df = backtest_predictions(documents, cache, quotes_df, max_prev)
        if not results_df.empty:
            results_df = results_df.rename(columns={'cumulative_next_bar_pips': f'max_{max_prev}'})
            if all_results.empty:
                all_results = results_df
            else:
                all_results = all_results.merge(results_df, on='test_date', how='outer')

    # Сохранение в XLSX
    if not all_results.empty:
        all_results.to_excel(output_file, index=False, engine='openpyxl')
        logger.info(f"Результаты сохранены в {output_file}")
    else:
        logger.error("Нет результатов для сохранения.")

if __name__ == '__main__':
    main()