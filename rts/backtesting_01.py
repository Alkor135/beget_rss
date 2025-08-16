"""
Скрипт для ретроспективного предсказания (backtesting) на основе markdown-файлов с новостями.
Кэширует эмбеддинги в pickle-файл для избежания повторного создания ChromaDB.
Проверяет актуальность кэша при изменении или добавлении markdown-файлов.
Ограничивает количество предыдущих файлов для предсказаний параметром max_prev_files.
Добавляет финансовый результат в пунктах (pips) и накопительный результат (cumulative_pips).
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

# Параметры
md_path = Path(r'C:\Users\Alkor\gd\md_rss_investing')
cache_file = Path(r'embeddings_cache.pkl')
path_db_quote = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_2025_21-00.db')
model_name = "bge-m3"
url_ai = "http://localhost:11434/api/embeddings"
min_prev_files = 5   # Минимальное количество предыдущих файлов для предсказаний
max_prev_files = 30  # Максимальное количество предыдущих файлов для предсказаний

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
    """Читает таблицу Futures из базы данных котировок и возвращает DataFrame с pips."""
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['pips'] = df['CLOSE'] - df['OPEN']
    return df[['TRADEDATE', 'pips']].set_index('TRADEDATE')

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
        print("Кэш устарел: изменился набор markdown-файлов.")
        return False

    # Проверяем, не изменились ли файлы
    for doc in documents:
        file_path = md_path / doc.metadata['source']
        if file_path.stat().st_mtime > cache_mtime:
            print(f"Кэш устарел: файл {file_path.name} был изменён.")
            return False

    return True

def cache_embeddings(documents, cache_file, model_name, url_ai):
    """Вычисляет и кэширует эмбеддинги всех документов в pickle-файл."""
    if cache_is_valid(documents, cache_file):
        print(f"Загрузка кэша эмбеддингов из {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        return cache

    print("Вычисление эмбеддингов...")
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
    print(f"Эмбеддинги сохранены в {cache_file}")
    return cache

def backtest_predictions(documents, cache, quotes_df):
    """Проводит backtesting: для каждой тестовой даты симулирует предсказание с использованием кэша."""
    results = []
    total_predictions = 0
    correct_predictions = 0

    for test_idx in range(min_prev_files, len(documents)):
        test_doc = documents[test_idx]
        real_next_bar = test_doc.metadata['next_bar']
        test_date = test_doc.metadata['date']

        if real_next_bar == 'unknown' or real_next_bar == 'None':
            print(f"Пропуск даты {test_date}: next_bar неизвестен. {real_next_bar=}")
            continue

        # Получение эмбеддинга тестовой даты из кэша
        test_id = hashlib.md5(test_doc.page_content.encode()).hexdigest()
        test_embedding = None
        for item in cache:
            if item['id'] == test_id:
                test_embedding = item['embedding']
                break
        if test_embedding is None:
            print(f"Эмбеддинг для даты {test_date} не найден в кэше.")
            continue

        # Получение предыдущих документов из кэша, ближайших по дате
        prev_cache = sorted(
            [item for item in cache if item['metadata']['date'] < test_date],
            key=lambda x: x['metadata']['date'], reverse=True
        )[:max_prev_files]  # Ограничиваем max_prev_files ближайшими датами

        if len(prev_cache) < min_prev_files:
            print(f"Недостаточно предыдущих документов для даты {test_date}: {len(prev_cache)}")
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

            # Получение pips из базы котировок
            try:
                pips_value = quotes_df.loc[test_date, 'pips']
                pips = abs(pips_value) if is_correct else -abs(pips_value)
            except KeyError:
                print(f"Данные котировок для даты {test_date} не найдены.")
                continue

            results.append({
                'test_date': test_date,
                'predicted_next_bar': predicted_next_bar,
                'real_next_bar': real_next_bar,
                'similarity': closest_similarity,
                'is_correct': is_correct,
                'pips': pips
            })

            total_predictions += 1
            if is_correct:
                correct_predictions += 1

            print(f"Дата: {test_date}, Предсказание: {predicted_next_bar}, Реальное: {real_next_bar}, "
                  f"Сходство: {closest_similarity:.2f}%, Правильно: {is_correct}, Pips: {pips}")

    # Создание DataFrame и добавление накопительного результата
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['cumulative_pips'] = results_df['pips'].cumsum()
    else:
        print("Нет результатов для сохранения.")

    # Статистика
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nОбщая точность: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        if not results_df.empty:
            print(f"Итоговый накопительный результат: {results_df['cumulative_pips'].iloc[-1]:.2f} пунктов")
    else:
        print("Нет предсказаний для оценки.")

    # Сохранение результатов в CSV
    results_df.to_csv('backtest_results.csv', index=False)
    print("Результаты сохранены в backtest_results.csv")

if __name__ == '__main__':
    # Загрузка котировок
    if not path_db_quote.exists():
        print(f"Ошибка: Файл базы данных котировок не найден: {path_db_quote}")
        exit(1)
    quotes_df = load_quotes(path_db_quote)

    # Загрузка markdown-файлов
    documents = load_markdown_files(md_path)
    if len(documents) < min_prev_files + 1:
        print(f"Недостаточно файлов: {len(documents)}. Требуется минимум {min_prev_files + 1}.")
        exit(1)

    # Кэширование эмбеддингов
    cache = cache_embeddings(documents, cache_file, model_name, url_ai)
    backtest_predictions(documents, cache, quotes_df)