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


def read_db_news_from_date(db_path_news: Path, date_min: str) -> pd.DataFrame:
    """
    Читает новости из базы данных начиная с указанной даты без верхней границы.
    """
    with sqlite3.connect(db_path_news) as conn:
        query = """
            SELECT * FROM news
            WHERE date > ?
        """
        return pd.read_sql_query(query, conn, params=(date_min,))


def save_titles_to_markdown(df_news: pd.DataFrame, file_path: Path, next_bar: str) -> None:
    """
    Сохраняет заголовки новостей в markdown-файл с метаданными о направлении следующей свечи.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        # Метаданные в формате markdown front matter
        file.write(f"---\nnext_bar: {next_bar}\n---\n\n")
        for _, row in df_news.iterrows():
            title = row['title']
            file.write(f"- {title}\n")  # Записываем только заголовок в файл


def save_latest_titles_to_markdown(db_path_news: Path, db_path_quote: Path,
                                   md_news_dir: Path) -> None:
    """
    Создает markdown-файл с заголовками новостей начиная с максимальной даты в базе котировок
    с 18:45 МСК и метаданными next_bar: current.
    """
    # Получаем максимальную дату из базы котировок
    df_quote = read_db_quote(db_path_quote)
    df_quote['TRADEDATE'] = pd.to_datetime(df_quote['TRADEDATE'])
    max_date = df_quote['TRADEDATE'].max()
    max_date_str = max_date.strftime("%Y-%m-%d")

    # Формируем начальную дату
    date_min = f"{max_date_str} 18:45:00"
    date_min_gmt = msk_to_gmt(date_min)

    # Читаем новости начиная с date_min_gmt
    df_news = read_db_news_from_date(db_path_news, date_min_gmt)

    if len(df_news) > 0:
        # Формируем имя файла
        file_name = f"current.md"
        file_path = md_news_dir / file_name

        # Сохраняем новости с метаданными next_bar: current
        save_titles_to_markdown(df_news, file_path, "current")


def main(path_db_quote: Path, path_db_news: Path, md_news_dir: Path) -> None:
    """
    Основная функция: читает котировки и новости, формирует и сохраняет markdown-файлы с новостями и метаданными.
    """
    df = read_db_quote(path_db_quote)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df.sort_values(by='TRADEDATE', inplace=True)
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['bar'] = df.apply(lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1)
    df['next_bar'] = df['bar'].shift(-1)
    df.dropna(inplace=True)

    for i in range(len(df) - 1, 0, -1):
        row1 = df.iloc[i]
        row2 = df.iloc[i - 1]

        file_name = f"{row1['TRADEDATE']}.md"
        date_max = f"{row1['TRADEDATE']} 18:45:00"
        date_min = f"{row2['TRADEDATE']} 18:45:00"
        date_max_gmt = msk_to_gmt(date_max)
        date_min_gmt = msk_to_gmt(date_min)

        print(f"{file_name} Дата max: {date_max}, Дата min: {date_min}")
        df_news = read_db_news(path_db_news, date_max_gmt, date_min_gmt)
        print(df_news)
        if len(df_news) == 0:
            break

        save_titles_to_markdown(df_news, Path(fr'{md_news_dir}/{file_name}'), row1['next_bar'])

    # Вызываем функцию для создания файла с последними новостями
    save_latest_titles_to_markdown(path_db_news, path_db_quote, md_news_dir)


if __name__ == '__main__':
    ticker = 'RTS'
    path_db_quote = Path(fr'c:\Users\Alkor\gd\data_quote_db\{ticker}_day_rss_2025.db')
    # path_db_quote = Path(fr'c:\Users\Alkor\gd\data_beget_rss\RTS_day_rss_2025.db')
    path_db_news = Path(fr'C:\Users\Alkor\gd\data_beget_rss\rss_news_investing.db')
    md_news_dir = Path('c:/Users/Alkor/gd/news_rss_md')

    if not path_db_quote.exists():
        print("Ошибка: Файл базы данных котировок не найден.")
        exit()

    if not path_db_news.exists():
        print("Ошибка: Файл базы данных новостей не найден.")
        exit()

    (Path(md_news_dir)).mkdir(parents=True, exist_ok=True)

    main(path_db_quote, path_db_news, md_news_dir)