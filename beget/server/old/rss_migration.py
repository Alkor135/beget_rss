"""
Этот скрипт предназначен для одноразовой миграции данных из старой единой БД в новые месячные БД.
После его запуска можно использовать второй скрипт для дальнейшего сбора данных —
он будет добавлять новости только в БД текущего месяца, а прошлые данные уже будут распределены по
своим файлам.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler

# Настройка логирования с ротацией по времени
log_handler = TimedRotatingFileHandler(
    '/home/user/rss_scraper/log/rss_migration.log',
    when='midnight',  # Новый файл каждый день в полночь
    interval=1,
    backupCount=3  # Хранить логи за 3 дней
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(log_handler)


def get_db_path(base_dir: str, year_month: str) -> str:
    """
    Формирует путь к файлу базы данных на основе года и месяца.
    """
    return os.path.join(base_dir, f"rss_news_investing_{year_month}.db")


def migrate_data(old_db_path: str, base_dir: str) -> None:
    """
    Миграция данных из старой единой БД в месячные БД.
    - Читает все записи из старой БД.
    - Группирует по году-месяцу.
    - Сохраняет в соответствующие месячные БД.
    - Удаляет дубликаты в каждой новой БД.
    """
    if not os.path.exists(old_db_path):
        logging.error(f"Старая БД не найдена: {old_db_path}")
        return

    try:
        with sqlite3.connect(old_db_path) as old_conn:
            df = pd.read_sql_query("SELECT date, title FROM news ORDER BY date", old_conn)

        if df.empty:
            logging.info("Старая БД пуста, миграция не требуется.")
            return

        # Преобразуем date в datetime для группировки
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)  # Удаляем записи с некорректными датами

        # Группируем по году-месяцу
        df['year_month'] = df['date'].dt.strftime('%Y_%m')
        grouped = df.groupby('year_month')

        for year_month, group_df in grouped:
            db_path = get_db_path(base_dir, year_month)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            with sqlite3.connect(db_path) as new_conn:
                try:
                    # Создаем таблицу, если не существует
                    new_conn.execute("""
                        CREATE TABLE IF NOT EXISTS news (
                            date TEXT,
                            title TEXT
                        )
                    """)
                    new_conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_news_date_title ON news(date, title)")

                    # Сохраняем данные для этого месяца (append, на случай если БД уже существует)
                    group_df[['date', 'title']].to_sql('news', new_conn, if_exists='append',
                                                       index=False)
                    logging.info(
                        f"Данные за {year_month} сохранены в {db_path}. Сохранено строк: {len(group_df)}")

                    # Удаляем дубликаты в этой БД
                    cursor = new_conn.execute("SELECT COUNT(*) FROM news")
                    before_count = cursor.fetchone()[0]
                    new_conn.execute("""
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
                    cursor = new_conn.execute("SELECT COUNT(*) FROM news")
                    after_count = cursor.fetchone()[0]
                    deleted_count = before_count - after_count
                    logging.info(f"Дубликаты в {db_path} удалены. Удалено строк: {deleted_count}")

                    # Выполняем VACUUM
                    new_conn.isolation_level = None
                    new_conn.execute("VACUUM")
                    logging.info(f"VACUUM выполнен для {db_path}: база данных оптимизирована.")
                except Exception as e:
                    logging.error(f"Ошибка при миграции в {db_path}: {e}")

        logging.info("Миграция завершена успешно.")

        # Опционально: Удалить старую БД после миграции (раскомментируйте, если нужно)
        # os.remove(old_db_path)
        # logging.info(f"Старая БД удалена: {old_db_path}")

    except Exception as e:
        logging.error(f"Ошибка при миграции данных: {e}")


if __name__ == '__main__':
    OLD_DB_PATH = "/home/user/rss_scraper/db_data/rss_news_investing.db"
    BASE_DIR = "/home/user/rss_scraper/db_data"
    migrate_data(OLD_DB_PATH, BASE_DIR)