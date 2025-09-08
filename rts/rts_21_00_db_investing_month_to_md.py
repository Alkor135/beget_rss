"""
Скрипт для чтения базы данных котировок и новостей из нескольких файлов БД, формирования markdown-файлов с заголовками новостей.
Сохраняет не более указанных последних интервалов в формате markdown с метаданными.
Использует базу данных котировок с дневными свечами, сформированными из минутных данных с 21:00 МСК предыдущей сессии.
Обрабатывает последние два файла БД новостей (по месяцам).
Удаляет только самый последний markdown-файл по дате перед генерацией.
Не перезаписывает существующие markdown-файлы.
"""

import pandas as pd
from pathlib import Path
import sqlite3
import logging

# Параметры
ticker: str = 'RTS'  # Тикер фьючерса
ticker_lc: str = 'rts'  # Тикер фьючерса в нижнем регистре
rss_provider: str = 'investing'  # Поставщик новостей
# Директория с БД дневных свечей с 21:00 предыдущей сессии до 21:00 даты свечи.
path_db_quote = Path(fr'C:/Users/Alkor/gd/data_quote_db/{ticker}_futures_day_2025_21-00.db')
# Директория с файлами БД новостей по месяцам
db_news_dir = Path(fr'C:/Users/Alkor/gd/db_rss_{rss_provider}')
# Директория для сохранения markdown-файлов с новостями с 21:00 МСК предыдущей торговой сессии
md_news_dir = Path(fr'c:/Users/Alkor/gd/md_{ticker_lc}_{rss_provider}')
num_mds: int = 100  # Количество последних интервалов для сохранения в markdown файлы
num_dbs: int = 4  # Количество последних файлов БД новостей для обработки
time_start = '21:00:00'  # Время с которого начинается поиск новостей за предыдущую сессию в БД
time_end = '20:59:59'  # Время, которым заканчивается поиск новостей за текущую сессию в БД

# Настройка логирования: вывод в консоль и в файл, файл перезаписывается
log_file = Path(
    fr'C:\Users\Alkor\gd\predict_ai\{ticker_lc}_investing_ollama\log\{ticker_lc}_21_00_db_investing_month_to_md.txt')
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Удаляем существующие обработчики, чтобы избежать дублирования
logger.handlers = []
logger.addHandler(logging.FileHandler(log_file))
# Обработчик для консоли
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
# Обработчик для файла (перезаписывается при каждом запуске)
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def read_db_quote(db_path_quote: Path) -> pd.DataFrame:
    """
    Читает таблицу Futures из базы данных дневных котировок и возвращает DataFrame.
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
        date_max: str) -> None:
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


def delete_latest_md_file(md_news_dir: Path) -> None:
    """
    Удаляет самый последний markdown-файл по дате (имя файла) в директории.
    """
    md_files = sorted(md_news_dir.glob("*.md"), key=lambda f: f.stem, reverse=True)  # Сортировка по дате descending
    if md_files:
        latest_file = md_files[0]
        latest_file.unlink()
        logger.info(f"Удалён самый последний markdown-файл: {latest_file}")


def main(
        path_db_quote: Path = path_db_quote,
        db_news_dir: Path = db_news_dir,
        md_news_dir: Path = md_news_dir,
        num_mds: int = 100,
        num_dbs: int = 3) -> None:
    """
    Основная функция: читает котировки и новости из последних num_dbs файлов БД,
    удаляет самый последний markdown-файл, формирует и сохраняет не более num_mds markdown-файлов
    с новостями и метаданными за самые последние даты, не перезаписывая существующие файлы.
    """
    # Создаем директорию для сохранения markdown-файлов, если она не существует
    md_news_dir.mkdir(parents=True, exist_ok=True)

    if not path_db_quote.exists():
        logger.error(f"Ошибка: Файл базы данных котировок не найден. {path_db_quote}")
        exit()

    # Проверяем наличие файлов БД новостей
    db_files = list(db_news_dir.glob("rss_news_investing_*_*.db"))
    if not db_files:
        logger.error("Ошибка: Файлы баз данных новостей не найдены.")
        exit()

    # Получаем последние файлы БД новостей
    db_paths = get_latest_db_files(db_news_dir, num_files=num_dbs)
    if len(db_paths) < num_dbs:
        logger.error(f"Предупреждение: Найдено только {len(db_paths)} файлов БД, ожидалось {num_dbs}")

    logger.info("Используемые файлы БД:", [str(p) for p in db_paths])

    # Удаляем только самый последний markdown-файл
    delete_latest_md_file(md_news_dir)

    # Читаем базу данных котировок и формируем DataFrame
    df = read_db_quote(path_db_quote)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df.sort_values(by='TRADEDATE', inplace=True)
    df = df.tail(num_mds + 1)  # Ограничиваем до num_mds+1 строки, чтобы получить num_mds интервалов
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['bar'] = df.apply(lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1)
    df['next_bar'] = df['bar'].shift(-1)

    for i in range(len(df) - 1, 0, -1):
        row1 = df.iloc[i]
        row2 = df.iloc[i - 1]

        file_name = f"{row1['TRADEDATE']}.md"  # Формирование имени файла из даты
        file_path = md_news_dir / file_name
        date_min = f"{row2['TRADEDATE']} {time_start}"  # Дата и время старта поиска новостей в БД
        date_max = f"{row1['TRADEDATE']} {time_end}"  # Дата и время окончания поиска новостей в БД

        if file_path.exists():
            logger.info(f"Файл {file_name} уже существует, пропускаем.")
            continue

        logger.info(f"{file_name}. Новости за период: {date_min} - {date_max}")
        df_news = read_db_news_multiple(db_paths, date_max, date_min)
        if len(df_news) == 0:
            break

        save_titles_to_markdown(
            df_news, file_path,
            row1['next_bar'], date_min, date_max
        )


if __name__ == '__main__':
    main(path_db_quote, db_news_dir, md_news_dir, num_mds=num_mds, num_dbs=num_dbs)