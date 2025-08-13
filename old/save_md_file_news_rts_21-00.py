"""
Скрипт для чтения базы данных котировок и новостей, формирования markdown-файлов с заголовками новостей.
Сохраняет не более 30 последних интервалов в формате markdown с метаданными.
Использует базу данных котировок с дневными свечами, сформированными из минутных данных с 21:00 МСК.
"""

import pandas as pd
from pathlib import Path
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo


def msk_to_gmt(dt_str: str) -> str:
    """
    Преобразует строку даты-времени из МСК в GMT (ISO-формат).
    """
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=ZoneInfo("Europe/Moscow"))
    dt_gmt = dt.astimezone(ZoneInfo("Etc/GMT"))
    return dt_gmt.strftime("%Y-%m-%d %H:%M:%S")


def read_db_quote(db_path_quote: Path) -> pd.DataFrame:
    """
    Читает таблицу Futures из базы данных котировок и возвращает DataFrame.
    """
    with sqlite3.connect(db_path_quote) as conn:
        return pd.read_sql_query("SELECT * FROM Futures", conn)


def read_db_news(db_path_news: Path, date_max: str, date_min: str) -> pd.DataFrame:
    """
    Читает новости из базы данных за указанный период времени.
    """
    with sqlite3.connect(db_path_news) as conn:
        query = """
            SELECT * FROM news
            WHERE date > ? AND date < ?
        """
        return pd.read_sql_query(query, conn, params=(date_min, date_max))

def save_titles_to_markdown(
        df_news: pd.DataFrame,
        file_path: Path,
        next_bar: str,
        date_min: str,
        date_max: str
    ) -> None:
    """
    Сохраняет заголовки новостей в markdown-файл с метаданными.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        # Метаданные в формате markdown front matter
        file.write(f"---\nnext_bar: {next_bar}\ndate_min: {date_min}\ndate_max: {date_max}\n---\n\n")
        for _, row in df_news.iterrows():
            title = row['title']
            file.write(f"- {title}\n")  # Записываем только заголовок в файл

def main(path_db_quote: Path, path_db_news: Path, md_news_dir: Path, number_of_files: int) -> None:
    """
    Основная функция: читает котировки и новости, удаляет старые markdown-файлы,
    формирует и сохраняет не более 30 markdown-файлов с новостями и метаданными за самые последние даты.
    """
    # Удаляем все старые markdown-файлы в директории
    for old_file in md_news_dir.glob("*.md"):
        if old_file.is_file():  # Проверяем, что это файл, а не папка
            old_file.unlink()

    # Читаем базу данных котировок и формируем DataFrame
    df = read_db_quote(path_db_quote)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df.sort_values(by='TRADEDATE', inplace=True)
    # print(df)
    df = df.tail(number_of_files + 1)  # Ограничиваем до 31 строки, чтобы получить 30 интервалов
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['bar'] = df.apply(lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1)
    df['next_bar'] = df['bar'].shift(-1)
    # df.dropna(inplace=True)

    for i in range(len(df) - 1, 0, -1):
        row1 = df.iloc[i]
        row2 = df.iloc[i - 1]

        file_name = f"{row1['TRADEDATE']}.md"
        date_max = f"{row1['TRADEDATE']} 21:00:00"
        date_min = f"{row2['TRADEDATE']} 21:00:00"
        date_max_gmt = msk_to_gmt(date_max)
        date_min_gmt = msk_to_gmt(date_min)

        print(f"{file_name}. Новости за период: {date_min} - {date_max}")
        df_news = read_db_news(path_db_news, date_max_gmt, date_min_gmt)
        if len(df_news) == 0:
            break

        save_titles_to_markdown(
            df_news, Path(fr'{md_news_dir}/{file_name}'),
            row1['next_bar'], date_min_gmt, date_max_gmt
        )

if __name__ == '__main__':
    ticker = 'RTS'
    path_db_quote = Path(fr'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db')
    path_db_news = Path(fr'C:\Users\Alkor\gd\data_beget_rss\rss_news_investing.db')
    md_news_dir = Path('c:/Users/Alkor/gd/news_rss_md_rts_21-00')
    number_of_files = 30  # Максимальное количество сохраняемых markdown-файлов

    # Создаем директорию для сохранения markdown-файлов, если она не существует
    (Path(md_news_dir)).mkdir(parents=True, exist_ok=True)

    if not path_db_quote.exists():
        print(f"Ошибка: Файл базы данных котировок не найден. {path_db_quote}")
        exit()

    if not path_db_news.exists():
        print("Ошибка: Файл базы данных новостей не найден.")
        exit()

    main(path_db_quote, path_db_news, md_news_dir, number_of_files)