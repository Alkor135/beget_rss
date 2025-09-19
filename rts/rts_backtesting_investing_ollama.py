"""
Скрипт для ретроспективного предсказания (backtesting) на основе markdown-файлов с новостями.
Кэширует эмбеддинги в pickle-файл для избежания повторного создания ChromaDB.
Проверяет актуальность кэша при изменении или добавлении markdown-файлов.
Ограничивает количество предыдущих файлов для предсказаний параметром max_prev_files.
Добавляет финансовый результат в пунктах (next_bar_pips) и
накопительный результат (cumulative_next_bar_pips).
Сохраняет результаты XLSX файл.
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
import datetime
import logging

# Параметры
ticker: str = 'RTS'  # Тикер фьючерса
ticker_lc: str = 'rts'  # Тикер фьючерса в нижнем регистре
md_path = Path(fr'C:\Users\Alkor\gd\md_{ticker_lc}_investing')
cache_file = Path(fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\{ticker_lc}_embeddings_investing_ollama.pkl')
path_db_quote = Path(fr'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db')
model_name = "bge-m3"
url_ai = "http://localhost:11434/api/embeddings"
min_prev_files = 4  # Минимальное количество предыдущих файлов для предсказаний
max_prev_files = 7  # Максимальное количество предыдущих файлов для предсказаний
# Итоговый XLSX файл
result_file = Path(
    fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\{ticker_lc}_backtest_results_investing_ollama.xlsx')

# Настройка логирования: вывод в консоль и в файл, файл перезаписывается
log_file = Path(
    fr'C:\Users\Alkor\gd\predict_ai\{ticker_lc}_investing_ollama\log\{ticker_lc}_backtesting_investing_ollama.txt')
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Удаляем существующие обработчики, чтобы избежать дублирования
logger.handlers = []
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
    # Получаем список всех .md файлов в указанной директории и сортируем их по имени файла (имя = дата)
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)  # Сортировка по дате ascending
    documents = []  # Список для хранения объектов Document
    for file_path in files:
        # Открываем файл и читаем его содержимое
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Проверяем, есть ли Front Matter (метаданные в формате YAML, ограниченные ---)
        if content.startswith('---'):
            # Разделяем контент на части: до метаданных, саму метадату и основной текст
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1].strip()  # Вторая часть — это YAML-метаданные
                text_content = parts[2].strip()  # Третья часть — это основной текст документа
                # Парсим YAML в словарь Python
                metadata = yaml.safe_load(metadata_yaml) or {}
                # Формируем новый словарь метаданных, преобразуя значения в строки и добавляя дополнительную информацию
                metadata_str = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),  # Направление следующего бара (up/down/None(для текущей даты))
                    "date_min": str(metadata.get("date_min", "unknown")),  # Минимальная дата
                    "date_max": str(metadata.get("date_max", "unknown")),  # Максимальная дата
                    "source": file_path.name,  # Имя файла (исходник)
                    "date": file_path.stem  # Имя файла без расширения (предполагается, что это дата)
                }
                # Создаём объект Document с текстом и метаданными и добавляем его в список
                doc = Document(page_content=text_content, metadata=metadata_str)
                documents.append(doc)
    return documents  # Возвращаем список документов

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

def main(max_prev_files: int = 8):
    """Проводит backtesting: для каждой тестовой даты симулирует предсказание с использованием кэша."""
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

    # Кэширование эмбеддингов
    cache = cache_embeddings(documents, cache_file, model_name, url_ai)

    results = []
    total_predictions = 0
    correct_predictions = 0

    for test_idx in range(min_prev_files, len(documents)):
        test_doc = documents[test_idx]
        real_next_bar = test_doc.metadata['next_bar']
        test_date = test_doc.metadata['date']

        if real_next_bar == 'unknown' or real_next_bar == 'None':
            logger.info(f"Пропуск даты {test_date}: next_bar неизвестен. {real_next_bar=}")
            continue

        # Получение эмбеддинга тестовой даты из кэша
        test_id = hashlib.md5(test_doc.page_content.encode()).hexdigest()
        test_embedding = None
        for item in cache:
            if item['id'] == test_id:
                test_embedding = item['embedding']
                break
        if test_embedding is None:
            logger.error(f"Эмбеддинг для даты {test_date} не найден в кэше.")
            continue

        # Преобразование даты в формат datetime
        try:
            test_date_dt = datetime.datetime.strptime(test_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Некорректный формат даты: {test_date}")
            return

        # Получение предыдущих документов из кэша
        prev_cache = sorted(
            [item for item in cache if
             item['metadata']['next_bar'] != "None" and
             datetime.datetime.strptime(item['metadata']['date'], '%Y-%m-%d') < test_date_dt],
            key=lambda x: datetime.datetime.strptime(x['metadata']['date'], '%Y-%m-%d'),
            reverse=True
        )[:max_prev_files]  # Ограничиваем max_prev_files предыдущими датами

        # # Получение предыдущих документов из кэша, ближайших по дате
        # prev_cache = sorted(
        #     [item for item in cache if item['metadata']['date'] < test_date],
        #     key=lambda x: x['metadata']['date'], reverse=True
        # )[:max_prev_files]  # Ограничиваем max_prev_files ближайшими датами

        if len(prev_cache) < min_prev_files:
            logger.error(f"Недостаточно предыдущих документов для даты {test_date}: {len(prev_cache)}")
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
                logger.error(f"Данные котировок для даты {test_date} не найдены.")
                continue

            results.append({
                'test_date': test_date,
                'predicted_next_bar': predicted_next_bar,
                'real_next_bar': real_next_bar,
                'similarity': closest_similarity,
                'is_correct': is_correct,
                'next_bar_pips': next_bar_pips
            })

            total_predictions += 1
            if is_correct:
                correct_predictions += 1

            logger.info(f"Дата: {test_date}, Предсказание: {predicted_next_bar}, Реальное: {real_next_bar}, "
                  f"Сходство: {closest_similarity:.2f}%, Правильно: {is_correct}, next_bar_pips: {next_bar_pips}")

    # Создание DataFrame и добавление накопительного результата
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['cumulative_next_bar_pips'] = results_df['next_bar_pips'].cumsum()
    else:
        logger.info("Нет результатов для сохранения.")

    # Статистика
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        logger.info(f"Общая точность: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        if not results_df.empty:
            logger.info(
                f"Итоговый накопительный для {min_prev_files}/{max_prev_files} результат: "
                f"{results_df['cumulative_next_bar_pips'].iloc[-1]:.2f} пунктов"
            )
    else:
        logger.info("Нет предсказаний для оценки.")

    # Сохранение результатов в CSV и XLSX
    if not results_df.empty:
        results_df.to_excel(result_file, index=False, engine='openpyxl')
        logger.info(f"Результаты сохранены в {result_file}")
    else:
        logger.error("Нет результатов для сохранения в файл.")

if __name__ == '__main__':
    main(max_prev_files=max_prev_files)