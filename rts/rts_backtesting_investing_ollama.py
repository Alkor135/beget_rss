"""
Скрипт для ретроспективного предсказания (backtesting) на основе markdown-файлов с новостями.
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
import sqlite3
from langchain_core.documents import Document
from datetime import datetime
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
url_ai = settings['url_ai']  # Ollama API без тайм-аута
model_name = settings['model_name']  # Ollama модель
min_prev_files = settings['min_prev_files']  # Минимальное количество предыдущих файлов
max_prev_files = settings['max_prev_files']  # Максимальное количество предыдущих файлов

md_path = Path(  # Путь к markdown-файлам
    settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(
    settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
output_dir = Path(  # Путь к папке с результатами
    settings['output_dir'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
log_file = Path(  # Путь к файлу лога
    output_dir / 'log' / # Папка для логов
    fr'{ticker_lc}_backtesting_{provider}_ollama.txt')
result_file = Path(  # Итоговый XLSX файл
    fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}'
    fr'\{ticker_lc}_backtest_results_{provider}_ollama.xlsx')  # Путь к итоговому XLSX файлу

# Настройка логирования: вывод в консоль и в файл, файл перезаписывается
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

        # >>> Считаем md5 от всего содержимого файла (YAML + текст)
        md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        # Проверяем, есть ли Front Matter (метаданные в формате YAML, ограниченные ---)
        if content.startswith('---'):
            # Разделяем контент на части: до метаданных, саму метадату и основной текст
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1].strip()  # Вторая часть — это YAML-метаданные
                text_content = parts[2].strip()  # Третья часть — это основной текст документа
                # Парсим YAML в словарь Python
                metadata = yaml.safe_load(metadata_yaml) or {}
                # Формируем новый словарь метаданных
                metadata_str = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),  # Направление следующего бара (up/down/None(для текущей даты))
                    "date_min": str(metadata.get("date_min", "unknown")),  # Минимальная дата
                    "date_max": str(metadata.get("date_max", "unknown")),  # Максимальная дата
                    "source": file_path.name,  # Имя файла (исходник)
                    "date": file_path.stem,  # Имя файла без расширения (предполагается, что это дата)
                    "md5": md5_hash  # >>> Добавляем md5 для идентификации документа
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

def load_embeddings_from_cache(cache_file):
    """Загружает эмбеддинги только из готового pickle-файла.
    Если файл отсутствует — завершает выполнение с ошибкой.
    """
    if not cache_file.exists():
        logger.error(f"Ошибка: файл кэша {cache_file} не найден. "
                     f"Создание нового кэша запрещено для режима бэктестинга.")
        exit(1)

    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        logger.info(f"Эмбеддинги успешно загружены из {cache_file}")
        return cache
    except Exception as e:
        logger.error(f"Ошибка при загрузке кэша эмбеддингов: {e}")
        exit(1)


def main(max_prev_files: int = 8):
    """Проводит backtesting: использует только готовый pickle с эмбеддингами."""
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

    # ⚡ Загружаем эмбеддинги только из pkl (без пересоздания!)
    cache = load_embeddings_from_cache(cache_file)

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
        test_id = test_doc.metadata.get("md5")  # >>> Берём md5 прямо из метаданных, а не пересчитываем
        test_embedding = next((item['embedding'] for item in cache if item['id'] == test_id), None)

        if test_embedding is None:
            logger.error(f"Эмбеддинг для даты {test_date} (md5={test_id}) не найден в кэше.")
            continue

        # Преобразование даты
        try:
            test_date_dt = datetime.strptime(test_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Некорректный формат даты: {test_date}")
            return

        # Предыдущие документы по дате, не больше даты теста
        prev_cache = sorted(
            [item for item in cache if
             item['metadata']['next_bar'] != "None" and
             datetime.strptime(item['metadata']['date'], '%Y-%m-%d') < test_date_dt],
            key=lambda x: datetime.strptime(x['metadata']['date'], '%Y-%m-%d'),
            reverse=True
        )[:max_prev_files]

        if len(prev_cache) < min_prev_files:
            logger.error(f"Недостаточно предыдущих документов для даты {test_date}: {len(prev_cache)}")
            continue

        # Вычисляем сходства
        similarities = [
            (cosine_similarity(test_embedding, item['embedding']) * 100, item['metadata'])
            for item in prev_cache
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Ближайший по сходству документ
        if similarities:
            closest_similarity, closest_metadata = similarities[0]
            predicted_next_bar = closest_metadata['next_bar']
            is_correct = predicted_next_bar == real_next_bar

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

            logger.info(
                f"Дата: {test_date}, Предсказание: {predicted_next_bar}, "
                f"Реальное: {real_next_bar}, Сходство: {closest_similarity:.2f}%, "
                f"Правильно: {is_correct}, next_bar_pips: {next_bar_pips}"
            )

    # Сохраняем результаты
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['cumulative_next_bar_pips'] = results_df['next_bar_pips'].cumsum()

        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        logger.info(f"Общая точность: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        logger.info(
            f"Итоговый накопительный результат: "
            f"{results_df['cumulative_next_bar_pips'].iloc[-1]:.2f} пунктов"
        )

        results_df.to_excel(result_file, index=False, engine='openpyxl')
        logger.info(f"Результаты сохранены в {result_file}")
    else:
        logger.error("Нет результатов для сохранения.")

if __name__ == '__main__':
    main(max_prev_files=max_prev_files)