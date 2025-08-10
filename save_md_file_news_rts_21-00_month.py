"""
Скрипт для чтения базы данных котировок и новостей из нескольких файлов БД, формирования markdown-файлов с заголовками новостей.
Сохраняет не более 30 последних интервалов в формате markdown с метаданными.
Использует базу данных котировок с дневными свечами, сформированными из минутных данных с 21:00 МСК.
Обрабатывает последние три файла БД новостей (по месяцам).
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


def read_db_news_multiple(db_paths: list[Path], date_max: str, date_min: str) -> pd.DataFrame:
    """
    Читает новости из нескольких баз данных за указанный период времени.
    Использует ATTACH для подключения баз и UNION для объединения результатов.
    """
    conn = sqlite3.connect(':memory:')  # Временная in-memory БД для объединения
    for i, db_path in enumerate(db_paths):
        alias = f"db{i+1}"
        conn.execute(f"ATTACH DATABASE '{db_path}' AS {alias}")

    # Строим запрос с UNION ALL и фильтром по датам для каждой БД
    union_queries = []
    for i in range(len(db_paths)):
        alias = f"db{i+1}"
        union_queries.append(f"SELECT * FROM {alias}.news WHERE date > ? AND date < ?")

    full_query = " UNION ".join(union_queries)
    params = [date_min, date_max] * len(db_paths)  # Параметры для каждого подзапроса

    df = pd.read_sql_query(full_query, conn, params=params)

    # Отключаем базы
    for i in range(len(db_paths)):
        alias = f"db{i+1}"
        conn.execute(f"DETACH DATABASE {alias}")

    conn.close()
    return df


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


def get_latest_db_files(directory: Path, num_files: int = 3) -> list[Path]:
    """
    Находит последние num_files файлов БД новостей в директории, сортируя по году и месяцу в имени (descending).
    Фильтрует только файлы формата rss_news_investing_YYYY_MM.db.
    """
    files = []
    for f in directory.glob("rss_news_investing_*_*.db"):
        if f.is_file():
            parts = f.stem.split('_')
            if len(parts) >= 2:
                try:
                    year = int(parts[-2])
                    month = int(parts[-1])
                    files.append((f, year, month))
                except ValueError:
                    continue  # Пропускаем файлы с некорректным форматом

    # Сортировка по (год, месяц) descending
    files.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [f[0] for f in files[:num_files]]


def main(path_db_quote: Path, news_dir: Path, md_news_dir: Path, num_dbs: int = 3) -> None:
    """
    Основная функция: читает котировки и новости из последних num_dbs файлов БД,
    удаляет старые markdown-файлы, формирует и сохраняет не более 30 markdown-файлов
    с новостями и метаданными за самые последние даты.
    """
    # Получаем последние файлы БД новостей
    db_paths = get_latest_db_files(news_dir, num_files=num_dbs)
    if len(db_paths) < num_dbs:
        print(f"Предупреждение: Найдено только {len(db_paths)} файлов БД, ожидалось {num_dbs}")

    print("Используемые файлы БД:", [str(p) for p in db_paths])

    # Удаляем все старые markdown-файлы в директории
    for old_file in md_news_dir.glob("*.md"):
        if old_file.is_file():  # Проверяем, что это файл, а не папка
            old_file.unlink()

    # Читаем базу данных котировок и формируем DataFrame
    df = read_db_quote(path_db_quote)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df.sort_values(by='TRADEDATE', inplace=True)
    df = df.tail(31)  # Ограничиваем до 31 строки, чтобы получить 30 интервалов
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['bar'] = df.apply(lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1)
    df['next_bar'] = df['bar'].shift(-1)

    for i in range(len(df) - 1, 0, -1):
        row1 = df.iloc[i]
        row2 = df.iloc[i - 1]

        file_name = f"{row1['TRADEDATE']}.md"
        date_max = f"{row1['TRADEDATE']} 21:00:00"
        date_min = f"{row2['TRADEDATE']} 21:00:00"
        date_max_gmt = msk_to_gmt(date_max)
        date_min_gmt = msk_to_gmt(date_min)

        print(f"{file_name}. Новости за период: {date_min} - {date_max}")
        df_news = read_db_news_multiple(db_paths, date_max_gmt, date_min_gmt)
        if len(df_news) == 0:
            break

        save_titles_to_markdown(
            df_news, Path(fr'{md_news_dir}/{file_name}'),
            row1['next_bar'], date_min_gmt, date_max_gmt
        )


if __name__ == '__main__':
    ticker = 'RTS'
    path_db_quote = Path(fr'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db')
    news_dir = Path(fr'C:\Users\Alkor\gd\data_beget_rss')  # Директория с файлами БД новостей
    md_news_dir = Path('c:/Users/Alkor/gd/news_rss_md_rts_21-00_month')

    # Создаем директорию для сохранения markdown-файлов, если она не существует
    (Path(md_news_dir)).mkdir(parents=True, exist_ok=True)

    if not path_db_quote.exists():
        print(f"Ошибка: Файл базы данных котировок не найден. {path_db_quote}")
        exit()

    # Проверяем наличие файлов БД новостей
    db_files = list(news_dir.glob("rss_news_investing_*_*.db"))
    if not db_files:
        print("Ошибка: Файлы баз данных новостей не найдены.")
        exit()

    main(path_db_quote, news_dir, md_news_dir)