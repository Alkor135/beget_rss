# rss_scraper_all_providers.py
"""
Единый RSS-скрапер для Interfax, 1Prime и Investing.
Все новости сохраняются в одну SQLite-базу с указанием провайдера и временем загрузки.
База данных создаётся по месяцам (формат: rss_news_YYYY_MM.db).
Логирование в файл и консоль.
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import sqlite3
import os
import logging
from logging.handlers import RotatingFileHandler
from pytz import timezone
import glob
from pathlib import Path
import yaml

# Путь к settings.yaml
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

def get_db_path_by_date(base_dir, date_str):
    """
    Возвращает путь к БД по дате новости (формат YYYY-MM).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    db_name = f"rss_news_{dt.year}_{dt.month:02d}.db"
    return Path(base_dir) / db_name

# Параметры
URL_INVESTING = settings['url_investing']
URL_INTERFAX = settings['url_interfax']
URL_PRIME = settings['url_prime']

BASE_DIR = settings['base_dir']  # local path for testing
LOG_DIR = Path(Path(__file__).parent / 'log')  # server path
LOG_FILE = Path(LOG_DIR / "rss_scraper_all_providers.txt")  # server path

# Настройка логирования
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
# Настройка логирования: файл (1 МБ, 3 файла) + консоль
log_formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    style='%'
)
log_file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=1_048_576,  # 1 МБ
    backupCount=1,
    encoding='utf-8'
)
log_file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

root_logger = logging.getLogger('')
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()
root_logger.addHandler(log_file_handler)
root_logger.addHandler(console_handler)

# --- Функции для каждого провайдера ---
async def fetch_rss_interfax(session: aiohttp.ClientSession, rss_link: str) -> list[dict]:
    """
    Скрапинг новостей с Интерфакс (только категория 'Экономика').
    """
    news_items = []
    try:
        async with session.get(rss_link) as response:
            xml_content = await response.text()
            root = ET.fromstring(xml_content)
            for item in root.findall('.//item'):
                category = item.find('category').text if item.find('category') is not None else ""
                if category.strip() != "Экономика":
                    continue
                title = item.find('title').text if item.find('title') is not None else "Нет заголовка"
                pub_date_raw = item.find('pubDate').text if item.find('pubDate') is not None else None
                pub_date = None
                if pub_date_raw:
                    try:
                        dt_obj = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %z")
                        pub_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        logging.error(f"Ошибка парсинга даты '{pub_date_raw}': {e}")
                if pub_date:
                    news_items.append({
                        "date": pub_date,
                        "title": title,
                        "provider": "interfax"
                    })
    except Exception as e:
        logging.error(f"Ошибка при парсинге {rss_link}: {e}")
    return news_items

async def fetch_rss_prime(session: aiohttp.ClientSession, rss_link: str) -> list[dict]:
    """
    Скрапинг новостей с 1Prime (все категории).
    """
    news_items = []
    try:
        async with session.get(rss_link) as response:
            xml_content = await response.text()
            root = ET.fromstring(xml_content)
            for item in root.findall('.//item'):
                title = item.find('title').text if item.find('title') is not None else "Нет заголовка"
                pub_date_raw = item.find('pubDate').text if item.find('pubDate') is not None else None
                pub_date = None
                if pub_date_raw:
                    try:
                        dt_obj = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %z")
                        pub_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        logging.error(f"Ошибка парсинга даты '{pub_date_raw}': {e}")
                if pub_date:
                    news_items.append({
                        "date": pub_date,
                        "title": title,
                        "provider": "prime"
                    })
    except Exception as e:
        logging.error(f"Ошибка при парсинге {rss_link}: {e}")
    return news_items

def get_investing_links(url: str) -> list[str]:
    """
    Получение всех RSS-ссылок с Investing.com.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        news_section = soup.find('h2', string='Новости')
        if news_section:
            rss_column = news_section.find_parent('div', class_='rssColumn halfSizeColumn float_lang_base_2')
            if rss_column:
                rss_box = rss_column.find('ul', class_='rssBox')
                if rss_box:
                    list_items = rss_box.find_all('li')
                    rss_links = [
                        item.find('a')['href']
                        for item in list_items
                        if item.find('a') and item.find('a').get('href', '').endswith('.rss')
                    ]
                    return rss_links
    except Exception as e:
        logging.error(f"Ошибка при получении ссылок Investing: {e}")
    return []

async def fetch_rss_investing(session: aiohttp.ClientSession, rss_link: str) -> list[dict]:
    """
    Скрапинг одной RSS-ленты Investing.com.
    """
    news_items = []
    try:
        async with session.get(rss_link) as response:
            xml_content = await response.text()
            root = ET.fromstring(xml_content)
            for item in root.findall('.//item'):
                title = item.find('title').text if item.find('title') is not None else "Нет заголовка"
                pub_date_raw = item.find('pubDate').text if item.find('pubDate') is not None else None
                pub_date = None
                if pub_date_raw:
                    try:
                        dt_obj = pd.to_datetime(pub_date_raw, utc=True, errors="coerce")
                        dt_obj = dt_obj.tz_convert(timezone('Europe/Moscow'))
                        pub_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        logging.error(f"Ошибка парсинга даты '{pub_date_raw}': {e}")
                if pub_date:
                    news_items.append({
                        "date": pub_date,
                        "title": title,
                        "provider": "investing"
                    })
    except Exception as e:
        logging.error(f"Ошибка при парсинге {rss_link}: {e}")
    return news_items

# --- Асинхронный сбор всех новостей ---
async def gather_all_news():
    timeout = aiohttp.ClientTimeout(total=40)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        interfax_news = await fetch_rss_interfax(session, URL_INTERFAX)
        prime_news = await fetch_rss_prime(session, URL_PRIME)
        investing_links = get_investing_links(URL_INVESTING)
        investing_news = []
        if investing_links:
            tasks = [fetch_rss_investing(session, link) for link in investing_links]
            results = await asyncio.gather(*tasks)
            for sublist in results:
                investing_news.extend(sublist)

        logging.info(f"Интерфакс: {len(interfax_news)} новостей")
        logging.info(f"1Prime: {len(prime_news)} новостей")
        logging.info(f"Investing: {len(investing_news)} новостей")

        return interfax_news + prime_news + investing_news

# --- Работа с БД ---
def create_db(db_path: str):
    """
    Создаёт таблицу, если не существует, только с нужными полями.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news (
                loaded_at TEXT,
                date TEXT,
                title TEXT,
                provider TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date_title_provider ON news(date, title, provider)")


def load_existing_news(db_path: str) -> set:
    """
    Загружает существующие (date, title, provider) для проверки дублей.
    """
    if not os.path.exists(db_path):
        return set()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT date, title, provider FROM news")
            return set(cursor.fetchall())
    except Exception as e:
        logging.error(f"Ошибка при загрузке существующих новостей: {e}")
        return set()

def save_to_sqlite(news_list: list[dict], base_dir: str):
    """
    Сохраняет новости в SQLite по месяцам (через executemany для скорости).
    """
    if not news_list:
        logging.info("Нет новостей для сохранения.")
        return

    from collections import defaultdict
    news_by_month = defaultdict(list)
    for item in news_list:
        db_path = get_db_path_by_date(base_dir, item["date"])
        news_by_month[db_path].append(item)

    for db_path, items in news_by_month.items():
        create_db(db_path)
        existing = load_existing_news(db_path)
        now_str = datetime.now(timezone('Europe/Moscow')).strftime("%Y-%m-%d %H:%M:%S")
        filtered = [
            (now_str, item["date"], item["title"], item["provider"])
            for item in items
            if (item["date"], item["title"], item["provider"]) not in existing
        ]
        if not filtered:
            logging.info(f"Все новости уже есть в базе {db_path.name}, ничего не добавлено.")
            continue

        with sqlite3.connect(db_path) as conn:
            try:
                conn.executemany(
                    "INSERT INTO news (loaded_at, date, title, provider) VALUES (?, ?, ?, ?)",
                    filtered
                )
                conn.commit()
                logging.info(f"Сохранено новостей: {len(filtered)} в {db_path.name}")
            except Exception as e:
                logging.error(f"Ошибка при сохранении в БД {db_path.name}: {e}")

def remove_duplicates_from_db(db_path: str):
    """
    Удаляет дубликаты по дате (без времени), title и provider.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM news")
            before_count = cursor.fetchone()[0]
            conn.execute("""
                DELETE FROM news
                WHERE rowid NOT IN (
                    SELECT rowid
                    FROM (
                        SELECT
                            rowid,
                            DATE(date) AS news_date,
                            title,
                            provider,
                            ROW_NUMBER() OVER (PARTITION BY DATE(date), title, provider ORDER BY date ASC) AS rn
                        FROM news
                    ) AS subquery
                    WHERE rn = 1
                );
            """)
            cursor = conn.execute("SELECT COUNT(*) FROM news")
            after_count = cursor.fetchone()[0]
            deleted_count = before_count - after_count
            logging.info(f"Дубликаты удалены. Удалено: {deleted_count}")
            conn.isolation_level = None
            conn.execute("VACUUM")
            logging.info("VACUUM выполнен для базы данных.\n")
    except Exception as e:
        logging.error(f"Ошибка при удалении дубликатов: {e}")

# --- Основная функция ---
def main():
    try:
        logging.info(f"Запуск сбора данных: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        all_news = asyncio.run(gather_all_news())
        if all_news:
            all_news.sort(key=lambda x: x["date"])
            save_to_sqlite(all_news, BASE_DIR)
            for db_file in Path(BASE_DIR).glob("rss_news_*.db"):
                remove_duplicates_from_db(db_file)
        else:
            logging.info("Нет новостей для обработки.")
    except Exception as e:
        logging.exception(f"Неожиданная ошибка в main(): {e}")

if __name__ == '__main__':
    main()
