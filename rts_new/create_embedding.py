"""
Скрипт для создания и обновления кэша эмбеддингов markdown-файлов.
Кэширует эмбеддинги в pickle-файл, обновляет только новые/изменённые файлы.
"""

from pathlib import Path
import pickle
import hashlib
import numpy as np
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import logging
import yaml
from datetime import datetime

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
ticker_lc = ticker.lower()
url_ai = settings.get('url_ai', 'http://localhost:11434/api/embeddings')  # Ollama API без тайм-аута
model_name = settings.get('model_name', 'bge-m3')  # Ollama модель
md_path = Path(settings['md_path'])  # Путь к markdown-файлам

# Путь к pkl-файлу с кэшем
cache_file = Path(settings['cache_file'].replace('{ticker_lc}', ticker_lc))

# Создание папки для логов
log_dir = Path(__file__).parent / 'log'
log_dir.mkdir(parents=True, exist_ok=True)

# Имя файла лога с датой и временем запуска (один файл на запуск!)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f'create_embedding_ollama_{timestamp}.txt'

# Настройка логирования: ТОЛЬКО один файл + консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # один файл
        logging.StreamHandler()                           # консоль
    ]
)

# Ручная очистка старых логов (оставляем только 3 самых новых)
def cleanup_old_logs(log_dir: Path, max_files: int = 3):
    """Удаляет старые лог-файлы, оставляя max_files самых новых."""
    log_files = sorted(log_dir.glob("create_embedding_ollama_*.txt"))
    if len(log_files) > max_files:
        for old_file in log_files[:-max_files]:
            try:
                old_file.unlink()
                print(f"Удалён старый лог: {old_file.name}")
            except Exception as e:
                print(f"Не удалось удалить {old_file}: {e}")

# Вызываем очистку ПЕРЕД началом логирования
cleanup_old_logs(log_dir, max_files=3)
logging.info(f"🚀 Запуск скрипта. Лог-файл: {log_file}")

def main():
    """Основная функция создания эмбеддингов."""
    # Проверяем существование папки с markdown файлами
    if not md_path.exists():
        logging.error(f"Папка с markdown файлами не найдена: {md_path}")
        return

    # Загрузка markdown-файлов
    documents = load_markdown_files(md_path)
    if not documents:
        logging.error("Не удалось загрузить markdown файлы")
        return

    # Создание/обновление кэша эмбеддингов
    cache = cache_embeddings(documents, cache_file, model_name, url_ai)
    logging.info("Создание эмбеддингов завершено успешно")


if __name__ == '__main__':
    main()
