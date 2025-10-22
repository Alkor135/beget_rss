"""
Вспомогательный скрипт для проверки кэша эмбеддингов с цветовой маркировкой:
- Загружает pkl-файл с эмбеддингами
- Выводит список md5 → source → date → next_bar
- Проверяет, что даты из кэша совпадают с датами дневных баров в SQLite3
- Проверяет, что md5 из кэша совпадают с md5 текущих .md файлов
✅ — зелёный (совпадение)
⚠️ — жёлтый (несоответствие)
"""

import pickle
import sqlite3
from pathlib import Path
import hashlib
import logging
import yaml

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings.get('ticker', "RTS")  # Тикер инструмента
ticker_lc = ticker.lower()
provider = settings.get('provider', 'investing')  # Провайдер RSS новостей
cache_file = Path(  # Путь к кэшу
    settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
md_path = Path(  # Путь к markdown-файлам
    settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))

# ==== Цвета ANSI ====
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# ==== Логирование ====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_cache(cache_file: Path):
    """Загружает кэш эмбеддингов из pkl."""
    if not cache_file.exists():
        logger.error(f"Файл кэша не найден: {cache_file}")
        exit(1)

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    logger.info(f"Кэш загружен: {len(cache)} записей")
    return cache


def load_db_dates(path_db_quote: Path):
    """Читает даты из SQLite базы (TRADEDATE из Futures)."""
    if not path_db_quote.exists():
        logger.error(f"Файл базы не найден: {path_db_quote}")
        exit(1)

    with sqlite3.connect(path_db_quote) as conn:
        rows = conn.execute("SELECT DISTINCT TRADEDATE FROM Futures ORDER BY TRADEDATE").fetchall()

    db_dates = {str(r[0]) for r in rows}
    logger.info(f"Из базы котировок загружено {len(db_dates)} уникальных дат")
    return db_dates


def load_md_files(md_path: Path):
    """Загружает список .md файлов и пересчитывает их md5."""
    md_files = {}
    if not md_path.exists():
        logger.error(f"Папка с MD-файлами не найдена: {md_path}")
        exit(1)

    for file_path in md_path.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            md5_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            md_files[file_path.name] = md5_hash
    logger.info(f"Найдено {len(md_files)} md-файлов")
    return md_files


def main():
    # Загружаем кэш, БД и MD-файлы
    cache = load_cache(cache_file)
    db_dates = load_db_dates(path_db_day)
    md_files = load_md_files(md_path)

    # Вывод содержимого кэша
    cache_dates = {item["metadata"]["date"] for item in cache}
    print("\n=== Содержимое кэша (md5 → source → date → next_bar) ===")
    for item in cache:
        md5 = item["id"]
        embedding = item["embedding"]
        meta = item["metadata"]
        print(f"{GREEN if md5 in md_files.values() else YELLOW}"
              f"{md5} → {meta['source']} → {meta['date']} → next_bar={meta['next_bar']} embedding={embedding[:3]}{RESET}")

    # Проверка соответствия дат
    missing_in_db = sorted(cache_dates - db_dates)
    missing_in_cache = sorted(db_dates - cache_dates)
    print("\n=== Проверка дат дневных баров ===")
    if missing_in_db:
        print(f"{YELLOW}⚠️ Даты есть в кэше, но отсутствуют в БД:{RESET}")
        for d in missing_in_db:
            print(f"  - {d}")
    else:
        print(f"{GREEN}✅ Все даты из кэша ембеддингов найдены в БД котировок{RESET}")

    if missing_in_cache:
        print(f"{YELLOW}⚠️ Даты есть в БД котировок, но отсутствуют в кэше ембеддингов:{RESET}")
        for d in missing_in_cache:
            print(f"  - {d}")
    else:
        print(f"{GREEN}✅ Все даты из БД найдены в кэше{RESET}")

    # ==== Проверка соответствия md5 из кэша и файлов ====
    print("\n=== Проверка соответствия MD5 файлов и кэша ===")
    cache_md5_map = {item["metadata"]["source"]: item["id"] for item in cache}

    for fname, md5 in md_files.items():
        if fname in cache_md5_map:
            if md5 == cache_md5_map[fname]:
                print(f"{GREEN}✅ {fname}: md5 совпадает{RESET}")
            else:
                print(f"{YELLOW}⚠️ {fname}: md5 НЕ совпадает (кэш={cache_md5_map[fname]}, файл={md5}){RESET}")
        else:
            print(f"{YELLOW}⚠️ {fname}: отсутствует в кэше{RESET}")

    # Проверяем, есть ли в кэше файлы которых нет на диске
    for fname in cache_md5_map:
        if fname not in md_files:
            print(f"{YELLOW}⚠️ {fname}: есть в кэше, но отсутствует на диске{RESET}")


if __name__ == "__main__":
    main()
