"""
Скрипт для предсказания направления следующей свечи (next_bar) на основе markdown-файлов с новостями.
Кэширует ембеддинги в pickle-файл, обновляет только новые/изменённые файлы.
Предсказывает направление следующего бара для md файла с next_bar='None'.
Для предсказаний ограничивает количество предыдущих файлов параметрами min_prev_files и max_prev_files.
Предсказывает направление для документа с next_bar='None'.
Все предсказания сохраняются в текстовый файл в папке predict_ollama с именем {none_date}.txt.
Логи пишутся в файл и в консоль, файл лога перезаписывается при каждом запуске.
Ембеддинги создает для новых и измененных файлов markdown, для остальных файлов, берет из кэша.
Передача файлов для ембеддинга батчами по 1 файлу.
"""

from pathlib import Path
import pickle
import hashlib
import numpy as np
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from contextlib import redirect_stdout
import datetime
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
min_prev_files = settings['min_prev_files']  # Минимальное количество предыдущих файлов для предсказания
max_prev_files = settings['max_prev_files']  # Максимальное количество предыдущих файлов для предсказания

md_path = Path(  # Путь к markdown-файлам
    settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(  # Путь к pkl-файлу с кэшем
    settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
output_dir = Path(  # Путь к папке с результатами
    settings['output_dir'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
log_file = Path(  # Путь к файлу лога
    output_dir / 'log' / # Папка для логов
    fr'{ticker_lc}_predict_next_session_{provider}_ollama.txt')  # Имя файла лога

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
    """Оптимизированная версия вычисления косинусного сходства с float64."""
    vec1 = np.asarray(vec1, dtype=np.float64)
    vec2 = np.asarray(vec2, dtype=np.float64)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return 0.0 if norm1 == 0 or norm2 == 0 else np.dot(vec1, vec2) / (norm1 * norm2)

def parse_metadata(content, file_path):
    """Извлекает метаданные и контент из markdown-файла."""
    # Проверяем, начинается ли содержимое файла с трёх дефисов ('---'),
    # что обычно используется в формате YAML для разделения метаданных и основного текста.
    if content.startswith('---'):
        # Делим содержимое файла на три части по символу '---'.
        # Ожидаем, что первая часть — пустая, вторая — YAML-метаданные, третья — текст.
        parts = content.split('---', 2)
        # Проверяем, что разбиение дало как минимум 3 части (начало, метаданные, текст).
        if len(parts) >= 3:
            # Вторая часть содержит YAML-метаданные, удаляем лишние пробелы.
            metadata_yaml = parts[1].strip()
            # Третья часть — основной текст документа.
            text_content = parts[2].strip()
            # Парсим YAML-строку в словарь Python.
            metadata = yaml.safe_load(metadata_yaml) or {}
            # Возвращаем текст и подготовленные метаданные.
            return text_content, {
                # Значение "next_bar" из метаданных, или "unknown", если не найдено.
                "next_bar": str(metadata.get("next_bar", "unknown")),
                # Минимальная дата из метаданных.
                "date_min": str(metadata.get("date_min", "unknown")),
                # Максимальная дата из метаданных.
                "date_max": str(metadata.get("date_max", "unknown")),
                # Имя файла как источник данных.
                "source": file_path.name,
                # Имя файла без расширения в качестве даты.
                "date": file_path.stem
            }
    else:
        # Если файл не начинается с '---', выводим ошибку.
        logger.error(f"Файл {file_path} не содержит метаданных.")
        # Прерываем выполнение программы.
        exit(0)

def load_markdown_files(directory):
    """Загружает все MD-файлы из директории, сортирует по дате."""
    try:
        # Получаем список всех .md файлов в указанной директории и сортируем их по имени файла (имя = дата).
        files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
        documents = []  # Создаём пустой список для хранения объектов Document langchain.
        for file_path in files:
            try:
                # Открываем markdown-файл в режиме чтения с кодировкой UTF-8.
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                # Вычисляем MD5-хэш содержимого файла
                md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                logger.info(f"MD5 хэш файла {file_path.name}: {md5_hash}")
                # Извлекаем текст и метаданные из содержимого с помощью функции parse_metadata.
                text_content, metadata = parse_metadata(content, file_path)
                # Сохраняем md5 в метаданные для дальнейшей проверки изменений
                metadata["md5"] = md5_hash
                # Создаём объект Document и добавляем его в список.
                documents.append(Document(page_content=text_content, metadata=metadata))
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {file_path}: {str(e)}")
                continue
        logger.info(f"Загружено {len(documents)} markdown-файлов")
        return documents  # Возвращаем список объектов Document.
    except Exception as e:
        logger.error(f"Ошибка при загрузке файлов из {directory}: {str(e)}")
        return []

def cache_embeddings(documents, cache_file, model_name, url_ai):
    """Кэширует эмбеддинги новых и изменённых файлов."""
    ef = OllamaEmbeddingFunction(model_name=model_name, url=url_ai)

    # Текущее состояние файлов (ключ = имя файла)
    current_files = {
        doc.metadata['source']: (doc, doc.metadata["md5"])
        for doc in documents
    }

    cache = []
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)

            # Фильтруем кэш: оставляем только те записи, которые не изменились
            cache = [
                item for item in cache
                if item['metadata']['source'] in current_files
                and item['metadata'].get("md5") == current_files[item['metadata']['source']][1]
            ]
            logger.info(f"Загружен кэш с {len(cache)} записями")
        except Exception as e:
            logger.error(f"Ошибка при загрузке кэша из {cache_file}: {str(e)}")
            cache = []

    cached_sources_md5 = { (item['metadata']['source'], item['metadata'].get("md5")) for item in cache }

    # Новые или изменённые документы
    new_docs = [
        doc for doc in documents
        if (doc.metadata['source'], doc.metadata["md5"]) not in cached_sources_md5
    ]

    if new_docs:
        logger.info(f"Вычисление эмбеддингов для {len(new_docs)} новых/изменённых файлов")
        batch_size = 1
        for i in range(0, len(new_docs), batch_size):
            batch_docs = new_docs[i:i + batch_size]
            batch_contents = [doc.page_content for doc in batch_docs]
            logger.info(f"Обработка батча {i // batch_size + 1} с {len(batch_contents)} документами")
            try:
                embeddings = ef(batch_contents)
                for j, doc in enumerate(batch_docs):
                    embedding = np.array(embeddings[j], dtype=np.float64)
                    logger.info(f"Эмбеддинг для {doc.metadata['source']}: {embedding[:5]}")
                    cache.append({
                        'id': doc.metadata["md5"],  # теперь это md5 из метаданных
                        'embedding': embeddings[j],
                        'metadata': doc.metadata
                    })
            except Exception as e:
                logger.error(f"Ошибка при обработке батча {i // batch_size + 1}: {str(e)}")
                continue

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            logger.info(f"Кэш обновлён в {cache_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении кэша в {cache_file}: {str(e)}")

    return cache

def print_prediction(none_date, prev_files, predicted_next_bar, closest_similarity, closest_metadata, output_file):
    """Форматирует и выводит предсказание."""
    output = [
        f"Предсказание для даты: {none_date}, с параметрами: {min_prev_files}/{max_prev_files}:",
        f"Файлы для сравнения: {', '.join(prev_files)}",
        f"Предсказанное направление: {predicted_next_bar}",
        f"Процент сходства: {closest_similarity:.2f}%",
        "Метаданные ближайшего похожего документа:"
    ]
    # Добавляем все ключи из метаданных, включая md5
    for key in sorted(closest_metadata.keys()):
        output.append(f"  {key}: {closest_metadata[key]}")
    return output

def main(max_prev_files: int = 7):
    """Предсказывает направление next_bar для документа с next_bar='None'."""
    # Загрузка markdown-файлов
    documents = load_markdown_files(md_path)
    if len(documents) < min_prev_files + 1:
        logger.error(f"Недостаточно файлов: {len(documents)}. Требуется минимум {min_prev_files + 1}.")
        exit(1)

    # Кэширование эмбеддингов
    cache = cache_embeddings(documents, cache_file, model_name, url_ai)

    # Создание папки для вывода, если она не существует
    output_dir.mkdir(parents=True, exist_ok=True)

    # Поиск документа с next_bar="None"
    none_doc = next((doc for doc in documents if doc.metadata['next_bar'] == "None"), None)
    if not none_doc:
        logger.error("Документ с next_bar='None' не найден.")
        return

    none_date = none_doc.metadata['date']
    output_file = output_dir / f"{none_date}.txt"
    output_file.unlink(missing_ok=True)
    logger.info(f"Удален старый файл прогноза {output_file}, если он был.")

    # Получаем md5 документа напрямую из метаданных (id = md5)
    none_id = none_doc.metadata["md5"]

    # Ищем эмбеддинг в кэше
    none_embedding = next((item['embedding'] for item in cache if item['id'] == none_id), None)
    if none_embedding is None:
        logger.error(f"Эмбеддинг для даты {none_date} (md5={none_id}) не найден в кэше.")
        return

    # Преобразование none_date в datetime
    try:
        none_date_dt = datetime.datetime.strptime(none_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Некорректный формат даты: {none_date}")
        return

    # Получение предыдущих документов из кэша, ближайших по дате
    prev_cache = sorted(
        [item for item in cache if
         item['metadata']['next_bar'] != "None" and
         datetime.datetime.strptime(item['metadata']['date'], '%Y-%m-%d') < none_date_dt],
        key=lambda x: datetime.datetime.strptime(x['metadata']['date'], '%Y-%m-%d'),
        reverse=True
    )[:max_prev_files]

    if len(prev_cache) < min_prev_files:
        logger.error(f"Недостаточно предыдущих документов для даты {none_date}: {len(prev_cache)}")
        return

    # Вывод списка файлов для сравнения
    prev_files = [item['metadata']['source'] for item in prev_cache]

    # Вычисление сходств
    similarities = [
        (cosine_similarity(none_embedding, item['embedding']) * 100, item['metadata'])
        for item in prev_cache
    ]
    similarities.sort(key=lambda x: x[0], reverse=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                if similarities:
                    closest_similarity, closest_metadata = similarities[0]
                    predicted_next_bar = closest_metadata['next_bar']
                    for line in print_prediction(
                        none_date, prev_files, predicted_next_bar,
                        closest_similarity, closest_metadata, output_file
                    ):
                        print(line)
                else:
                    print("Нет похожих документов для предсказания.")
        logger.info(f"Результаты сохранены в {output_file}")

        if similarities:
            for line in print_prediction(
                none_date, prev_files, predicted_next_bar,
                closest_similarity, closest_metadata, output_file
            ):
                logger.info(line)
    except Exception as e:
        logger.error(f"Ошибка при записи в файл предсказания {output_file}: {str(e)}")

if __name__ == '__main__':
    main(max_prev_files=max_prev_files)