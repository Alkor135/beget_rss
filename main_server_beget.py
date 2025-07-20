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
from logging.handlers import TimedRotatingFileHandler

# Настройка логирования с ротацией по времени
log_handler = TimedRotatingFileHandler(
    '/home/user/rss_scraper/rss_scraper.log',
    when='midnight',  # Новый файл каждый день в полночь
    interval=1,
    backupCount=7  # Хранить логи за 7 дней
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(log_handler)

async def fetch_rss(session: aiohttp.ClientSession, rss_link: str) -> list[dict]:
    """
    Асинхронно получает и парсит одну RSS-ленту, возвращает список новостей.
    """
    news_items = []
    try:
        async with session.get(rss_link) as response:
            xml_content = await response.text()
            root = ET.fromstring(xml_content)
            channel = root.find('.//channel')
            channel_name = channel.find('title').text if channel is not None and channel.find('title') is not None else ""
            logging.info(f'Обработка канала: {channel_name}')
            for item in root.findall('.//item'):
                title = item.find('title').text if item.find('title') is not None else "Нет заголовка"
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else "Нет даты публикации"
                link = item.find('link').text if item.find('link') is not None else "Нет ссылки"
                news_items.append({
                    "date": pub_date,
                    "section": channel_name,
                    "title": title,
                    "link": link
                })
    except Exception as e:
        logging.error(f"Ошибка при парсинге {rss_link}: {e}")
    return news_items

async def async_parsing_news(rss_links: list[str]) -> pd.DataFrame:
    """
    Асинхронно парсит все RSS-ленты и возвращает DataFrame.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_rss(session, link) for link in rss_links]
        results = await asyncio.gather(*tasks)
    all_news = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_news, columns=["date", "section", "title", "link"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def get_links(url: str) -> list[str]:
    """
    Получение ссылок на новостные RSS.
    """
    try:
        response = requests.get(url)
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
                else:
                    logging.error("Список RSS не найден.")
            else:
                logging.error("Контейнер с новостями не найден.")
        else:
            logging.error("Раздел 'Новости' не найден.")
    except Exception as e:
        logging.error(f"Ошибка при получении ссылок: {e}")
    return []

def parsing_news(rss_links: list[str]) -> pd.DataFrame:
    """
    Обёртка для асинхронного парсинга, чтобы вызывать из синхронного кода.
    """
    return asyncio.run(async_parsing_news(rss_links))

def save_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """
    Сохраняет DataFrame c RSS-лентой новостей в SQLite базу данных.
    """
    if df.empty:
        logging.error("DataFrame пустой, нечего сохранять в БД.")
        return
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    date TEXT,
                    title TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date_title ON news(date, title)")
            df[["date", "title"]].to_sql('news', conn, if_exists='append', index=False)
            logging.info(f"Новости сохранены в базе данных. Сохранено строк: {len(df)}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении в БД: {e}")

def remove_duplicates_from_db(db_path: str) -> None:
    """
    Удаляет дубликаты из таблицы news по дате (без времени) и title, оставляя одну запись.
    Выводит количество удалённых строк и выполняет VACUUM.
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
                            ROW_NUMBER() OVER (PARTITION BY DATE(date), title ORDER BY date ASC) AS rn
                        FROM news
                    ) AS subquery
                    WHERE rn = 1
                );
            """)
            cursor = conn.execute("SELECT COUNT(*) FROM news")
            after_count = cursor.fetchone()[0]
            deleted_count = before_count - after_count
            logging.info(f"Дубликаты в базе данных удалены. Удалено строк: {deleted_count}")
    except Exception as e:
        logging.error(f"Ошибка при удалении дубликатов из БД: {e}")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.isolation_level = None
            conn.execute("VACUUM")
            logging.info("VACUUM выполнен: база данных оптимизирована.")
    except Exception as e:
        logging.error(f"Ошибка при выполнении VACUUM: {e}")

def main(url: str, db_path: str) -> None:
    """
    Основная функция для парсинга RSS и сохранения в БД.
    """
    try:
        logging.info(f"Запуск сбора данных: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rss_links = get_links(url)
        if not rss_links:
            logging.error("Не удалось получить ссылки на RSS ленты.")
            return
        logging.info('Ссылки на RSS ленты получены')
        df = parsing_news(rss_links)
        df = df.sort_values(by='date')  # Сортировка по date в ascending order
        save_to_sqlite(df, db_path)
        remove_duplicates_from_db(db_path)
    except Exception as e:
        logging.error(f"Ошибка в main: {e}")

if __name__ == '__main__':
    URL = "https://ru.investing.com/webmaster-tools/rss"
    DB_PATH = "/home/user/rss_scraper/rss_news_investing.db"
    main(URL, DB_PATH)
