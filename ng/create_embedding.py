"""
Скрипт для создания и обновления кэша эмбеддингов markdown-файлов.
Кэширует эмбеддинги в pickle-файл, обновляет только новые/изменённые файлы.
Не создаёт файлы предсказаний.
"""

from pathlib import Path
import pickle
import hashlib
import numpy as np
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
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

md_path = Path(  # Путь к markdown-файлам
    settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))

cache_file = Path(  # Путь к pkl-файлу с кэшем
    settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))

log_file = Path(__file__).parent / f"{ticker_lc}_create_embedding_{provider}_ollama.txt"

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


def parse_metadata(content, file_path):
    """Извлекает метаданные и контент из markdown-файла."""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata_yaml = parts[1].strip()
            text_content = parts[2].strip()
            metadata = yaml.safe_load(metadata_yaml) or {}
            return text_content, {
                "next_bar": str(metadata.get("next_bar", "unknown")),
                "date_min": str(metadata.get("date_min", "unknown")),
                "date_max": str(metadata.get("date_max", "unknown")),
                "source": file_path.name,
                "date": file_path.stem
            }
    logger.error(f"Файл {file_path} не содержит метаданных.")
    return "", {}


def load_markdown_files(directory):
    """Загружает все MD-файлы из директории, сортирует по дате."""
    try:
        files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
        documents = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                logger.info(f"MD5 хэш файла {file_path.name}: {md5_hash}")
                text_content, metadata = parse_metadata(content, file_path)
                metadata["md5"] = md5_hash
                documents.append(Document(page_content=text_content, metadata=metadata))
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {file_path}: {str(e)}")
                continue
        logger.info(f"Загружено {len(documents)} markdown-файлов")
        return documents
    except Exception as e:
        logger.error(f"Ошибка при загрузке файлов из {directory}: {str(e)}")
        return []


def cache_embeddings(documents, cache_file, model_name, url_ai):
    """Кэширует эмбеддинги новых и изменённых файлов."""
    ef = OllamaEmbeddingFunction(model_name=model_name, url=url_ai)

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

    cached_sources_md5 = {(item['metadata']['source'], item['metadata'].get("md5")) for item in
                          cache}

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
            logger.info(
                f"Обработка батча {i // batch_size + 1} с {len(batch_contents)} документами")
            try:
                embeddings = ef(batch_contents)
                for j, doc in enumerate(batch_docs):
                    embedding = np.array(embeddings[j], dtype=np.float64)
                    logger.info(f"Эмбеддинг для {doc.metadata['source']}: {embedding[:5]}")
                    cache.append({
                        'id': doc.metadata["md5"],  # id = md5 из метаданных
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
        logger.info(f"Кэш обновлён в {cache_file}, всего записей: {len(cache)}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении кэша в {cache_file}: {str(e)}")

    return cache


def main():
    """Основная функция создания эмбеддингов."""
    # Проверяем существование папки с markdown файлами
    if not md_path.exists():
        logger.error(f"Папка с markdown файлами не найдена: {md_path}")
        return

    # Загрузка markdown-файлов
    documents = load_markdown_files(md_path)
    if not documents:
        logger.error("Не удалось загрузить markdown файлы")
        return

    # Создание/обновление кэша эмбеддингов
    cache = cache_embeddings(documents, cache_file, model_name, url_ai)
    logger.info("Создание эмбеддингов завершено успешно")


if __name__ == '__main__':
    main()
