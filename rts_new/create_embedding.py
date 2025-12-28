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
url_ai = settings['url_ai']  # Ollama API без тайм-аута
model_name = settings['model_name']  # Ollama модель
md_path = Path(settings['md_path'])  # Путь к markdown-файлам

cache_file = Path(  # Путь к pkl-файлу с кэшем
    settings['cache_file'].replace('{ticker_lc}', ticker_lc))

(Path(Path(__file__).parent / 'log')).mkdir(parents=True, exist_ok=True)
log_file = Path(__file__).parent / 'log' / f"{ticker_lc}_create_embedding_ollama.txt"

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