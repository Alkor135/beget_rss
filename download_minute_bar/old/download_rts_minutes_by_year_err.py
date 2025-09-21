"""
Скрипт скачивает минутные данные по фьючерсу RTS с MOEX ISS API и сохраняет их
в отдельные SQLite-базы по годам (2014, 2015, ...).
- Если база за год пустая → качает с 1 января.
- Если в базе уже есть данные → продолжает с последней даты.
- Для наглядности используется progress bar (tqdm).
- Каждая годовая база оптимизируется VACUUM после заполнения.
"""

from pathlib import Path
import sqlite3
from datetime import datetime, timedelta, date, time
import requests
import pandas as pd
import logging
from tqdm import tqdm


# ==== Параметры ====
TICKER = "RTS"                       # Код базового актива
START_DATE = date(2014, 1, 1)        # Начальная дата загрузки
OUTPUT_DIR = Path(fr"C:\Users\Alkor\gd\data_quote_db")      # Папка для БД
LOG_FILE = Path(__file__).parent / "download_rts_minutes.log"
# LOG_FILE = OUTPUT_DIR / "download_rts_minutes.log"


# ==== Логирование ====
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


# ==== Вспомогательные функции ====
def request_moex(session, url, retries=5, timeout=10):
    """Функция запроса данных с повторными попытками"""
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Ошибка запроса {url} (попытка {attempt + 1}): {e}")
            if attempt == retries - 1:
                return None


def create_tables(connection: sqlite3.Connection) -> None:
    """Создание таблицы Futures"""
    with connection:
        connection.execute(
            """CREATE TABLE if not exists Futures (
                TRADEDATE   TEXT PRIMARY KEY UNIQUE NOT NULL,
                SECID       TEXT NOT NULL,
                OPEN        REAL NOT NULL,
                LOW         REAL NOT NULL,
                HIGH        REAL NOT NULL,
                CLOSE       REAL NOT NULL,
                VOLUME      INTEGER NOT NULL,
                LSTTRADE    DATE NOT NULL
            )"""
        )
    logger.info("Таблица Futures создана (если отсутствовала)")


def get_db_path(base_dir: Path, ticker: str, year: int) -> Path:
    """Формируем путь к БД для конкретного года"""
    db_dir = base_dir / str(year)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / f"{ticker}_minutes_{year}.db"


def get_minute_candles(session, ticker: str, start_date: date,
                       from_str: str = None, till_str: str = None) -> pd.DataFrame:
    """Получает все минутные данные по фьючерсу за указанную дату с учетом пагинации"""
    if from_str is None:
        from_str = datetime.combine(start_date, time(0, 0)).isoformat()
    if till_str is None:
        till_str = datetime.combine(start_date, time(23, 59, 59)).isoformat()

    all_data = []
    start = 0
    page_size = 500

    while True:
        url = (
            f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?"
            f"interval=1&from={from_str}&till={till_str}&start={start}"
        )
        j = request_moex(session, url)
        if not j or "candles" not in j or not j["candles"].get("data"):
            break

        data = [{k: r[i] for i, k in enumerate(j["candles"]["columns"])} for r in j["candles"]["data"]]
        if not data:
            break

        all_data.extend(data)
        start += page_size

        if len(data) < page_size:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "begin": "TRADEDATE",
        "open": "OPEN",
        "close": "CLOSE",
        "high": "HIGH",
        "low": "LOW",
        "volume": "VOLUME"
    })
    df["SECID"] = ticker
    df = df.dropna(subset=["OPEN", "LOW", "HIGH", "CLOSE", "VOLUME"])
    return df[["TRADEDATE", "SECID", "OPEN", "LOW", "HIGH", "CLOSE", "VOLUME"]].reset_index(drop=True)


def save_to_db(df: pd.DataFrame, connection: sqlite3.Connection) -> None:
    """Сохраняет DataFrame в таблицу Futures"""
    if df.empty:
        return
    try:
        with connection:
            df.to_sql("Futures", connection, if_exists="append", index=False)
    except sqlite3.Error as e:
        logger.error(f"Ошибка при сохранении данных в БД: {e}")


def get_future_date_results(session, start_date: date, end_date: date,
                            ticker: str, connection: sqlite3.Connection) -> None:
    """Загружает минутные данные для диапазона дат"""
    cursor = connection.cursor()

    # Если база не пустая → находим последнюю дату
    cursor.execute("SELECT MAX(DATE(TRADEDATE)) FROM Futures")
    max_date_in_db = cursor.fetchone()[0]
    if max_date_in_db:
        start_date = max(start_date, datetime.strptime(max_date_in_db, "%Y-%m-%d").date())

    total_days = (end_date - start_date).days + 1
    if total_days <= 0:
        logger.info("Данные за этот год уже есть полностью.")
        return

    with tqdm(total=total_days, desc=f"Загрузка {start_date.year}", unit="дн.") as pbar:
        current_date = start_date
        while current_date <= end_date:
            minute_df = get_minute_candles(session, ticker, current_date)
            if not minute_df.empty:
                minute_df["LSTTRADE"] = current_date
                save_to_db(minute_df, connection)
            current_date += timedelta(days=1)
            pbar.update(1)


def main():
    today_date = datetime.now().date()
    with requests.Session() as session:
        current_date = START_DATE
        while current_date <= today_date:
            year = current_date.year
            path_db = get_db_path(OUTPUT_DIR, TICKER, year)

            connection = sqlite3.connect(str(path_db), check_same_thread=True)
            create_tables(connection)

            logger.info(f"=== Загрузка данных за {year} в {path_db} ===")
            end_of_year = date(year, 12, 31)
            end_date = min(end_of_year, today_date)

            get_future_date_results(session, current_date, end_date, TICKER, connection)

            # Оптимизация БД
            connection.execute("VACUUM")
            connection.close()
            logger.info(f"Закрыто соединение с БД {path_db}")

            current_date = end_of_year + timedelta(days=1)


if __name__ == "__main__":
    main()
