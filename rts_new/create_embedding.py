"""
Скрипт для создания и обновления кэша эмбеддингов markdown-файлов.
Кэширует эмбеддинги в pickle-файл, обновляет только новые/изменённые файлы.
"""

from pathlib import Path
import pickle
import hashlib
import numpy as np
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import logging
import yaml
from datetime import datetime
import pandas as pd
import tiktoken

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

enc = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(enc.encode(text))

# === Функция для эмбеддингов через Ollama ===
ef = OllamaEmbeddingFunction(model_name=model_name)

def md5_of_file(path: Path) -> str:
    """Возвращает MD5-хэш содержимого файла."""
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    return md5_hash

def load_existing_cache(cache_file: Path) -> pd.DataFrame | None:
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            logging.info(f"Загружен существующий кэш: {cache_file}, строк: {len(df)}")
            return df
        except Exception as e:
            logging.error(f"Не удалось загрузить кэш {cache_file}: {e}")
    return None

def build_embeddings_df(md_dir: Path, existing_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Создаёт датафрейм с колонками:
    TRADEDATE (дата из имени файла YYYY-MM-DD.md),
    MD5_hash (md5 содержимого файла),
    VECTORS (эмбеддинг файла через OllamaEmbeddingFunction).
    """
    cache_lookup = {}

    if existing_df is not None and not existing_df.empty:
        cache_lookup = {
            row["TRADEDATE"]: {
                "MD5_hash": row["MD5_hash"],
                "VECTORS": row["VECTORS"],
            }
            for _, row in existing_df.iterrows()
        }

    records = []

    md_files = sorted(md_dir.glob("*.md"))
    logging.info(f"Найдено markdown-файлов: {len(md_files)}")

    for md_file in md_files:
        # Имя файла ожидается в формате YYYY-MM-DD.md
        try:
            tradedate_str = md_file.stem  # 'YYYY-MM-DD'
        except Exception as e:
            logging.error(f"Не удалось извлечь дату из имени файла {md_file.name}: {e}")
            continue

        try:
            text = md_file.read_text(encoding='utf-8')
        except Exception as e:
            logging.error(f"Ошибка чтения файла {md_file}: {e}")
            continue

        if not text.strip():
            logging.info(f"Пустой файл, пропуск: {md_file}")
            continue

        # MD5-хэш содержимого
        md5_hash = md5_of_file(md_file)

        cached = cache_lookup.get(tradedate_str)

        if cached and cached["MD5_hash"] == md5_hash:
            # === ФАЙЛ НЕ ИЗМЕНИЛСЯ ===
            records.append(
                {
                    "TRADEDATE": tradedate_str,
                    "MD5_hash": md5_hash,
                    "VECTORS": cached["VECTORS"],
                }
            )
            logging.info(f"{md_file.name}: без изменений, взято из кэша")
            continue

        # Разбиение на чанки по параграфам (сохраняет пустые строки как разделители)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0

        if model_name == 'bge-m3':
            max_chunk_tokens = 7000  # Для bge-m3 (8192 лимит минус запас)
        elif model_name == 'qwen3-embedding:0.6b':
            max_chunk_tokens = 30000  # Для qwen3-embedding:0.6b (32768 лимит минус запас)
        else:
            print('Проверь модель')

        for para in paragraphs:
            para_len = token_len(para)  # Примерно токены
            if current_len + para_len > max_chunk_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        logging.info(f"{md_file.name}: чанков={len(chunks)}, токенов~={current_len}")

        # Эмбеддинги чанков
        chunk_embeddings = []
        for chunk in chunks:
            try:
                emb = ef([chunk])[0]
                chunk_embeddings.append(emb)
            except Exception as e:
                logging.error(f"Ошибка чанка в {md_file}: {e}")

        if not chunk_embeddings:
            continue

        # === ПРОВЕРКА РАЗМЕРНОСТИ ===
        dims = {len(e) for e in chunk_embeddings}
        if len(dims) != 1:
            logging.error(
                f"Несовпадение размерностей эмбеддингов в {md_file.name}: {dims}"
            )
            continue

        # === УСРЕДНЕНИЕ ===
        embedding = np.mean(chunk_embeddings, axis=0).tolist()

        records.append(
            {
                "TRADEDATE": tradedate_str,
                "MD5_hash": md5_hash,
                "VECTORS": embedding,
            }
        )

    df = pd.DataFrame(records, columns=["TRADEDATE", "MD5_hash", "VECTORS"])
    logging.info(f"Создан датафрейм эмбеддингов, строк: {len(df)}")
    return df

if __name__ == "__main__":
    existing_df = load_existing_cache(cache_file)

    df_embeddings = build_embeddings_df(md_path, existing_df)

    print(len(df_embeddings))

    with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 120
    ):
        print("Датафрейм с эмбеддингами:")
        print(df_embeddings.head())
    print(len(df_embeddings['VECTORS'].iloc[0]))

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(df_embeddings, f)
        logging.info(f"Кэш обновлён в {cache_file}, всего записей: {len(df_embeddings)}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении кэша в {cache_file}: {str(e)}")
