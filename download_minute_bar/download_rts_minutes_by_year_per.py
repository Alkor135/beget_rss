"""
Скрипт скачивает минутные данные по фьючерсу RTS с MOEX ISS API и сохраняет их по годам в отдельные базы SQLite.
Каждый файл содержит данные только за один календарный год.
Если в БД уже есть данные, докачиваются только недостающие бары.
Комментарии и структура кода адаптированы для поддержки годовых файлов.
"""

from pathlib import Path
import sqlite3
from datetime import datetime, timedelta, date, time
import requests
import pandas as pd
import logging
import yaml

# ==== НАСТРОЙКИ ====
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# Основные параметры
ticker = settings.get('ticker', 'RTS')
ticker_lc = ticker.lower()
start_date_raw = settings.get('start_date_download_minutes', "2014-01-01")
if isinstance(start_date_raw, date):
    start_date_global = start_date_raw
else:
    start_date_global = datetime.strptime(str(start_date_raw), "%Y-%m-%d").date()

start_date_global = datetime.strptime(settings.get('start_date_download_minutes', "2014-01-01"), "%Y-%m-%d").date()
output_dir = Path(settings.get('output_dir', 'C:/Users/Alkor/gd/data_quote_db')).resolve()
provider = settings.get('provider', 'moex')

# ==== ЛОГИРОВАНИЕ ====
def init_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    # Сброс старых хендлеров
    logger.handlers = []
    # Лог в файл
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # Лог в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    return logger

# ==== ЗАПРОСЫ К MOEX ISS ====
def request_moex(session, url, logger, retries=5, timeout=10):
    """Многократная попытка скачать данные MOEX ISS API с логированием ошибки."""
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Ошибка запроса {url} (попытка {attempt + 1}): {e}")
    return None

def get_info_future(session, security, logger):
    """
    Запрос информации о фьючерсе: наименование и дата экспирации.
    """
    url = f'https://iss.moex.com/iss/securities/{security}.json'
    j = request_moex(session, url, logger)
    if not j:
        return pd.Series(["", "2130-01-01"])
    data = [{k: r[i] for i, k in enumerate(j['description']['columns'])} for r in j['description']['data']]
    df = pd.DataFrame(data)
    shortname = df.loc[df['name'] == 'SHORTNAME', 'value'].values[0] if 'SHORTNAME' in df['name'].values else ""
    lsttrade = (
        df.loc[df['name'] == 'LSTTRADE', 'value'].values[0]
        if 'LSTTRADE' in df['name'].values
        else df.loc[df['name'] == 'LSTDELDATE', 'value'].values[0]
        if 'LSTDELDATE' in df['name'].values
        else "2130-01-01"
    )
    return pd.Series([shortname, lsttrade])

def get_minute_candles(session, ticker: str, query_date: date, logger, from_str: str = None, till_str: str = None) -> pd.DataFrame:
    """
    Загрузка минутных баров по инструменту за 1 день.
    """
    if from_str is None:
        from_str = datetime.combine(query_date, time(0, 0)).isoformat()
    if till_str is None:
        till_str = datetime.combine(query_date, time(23, 59, 59)).isoformat()
    all_data = []
    start = 0
    page_size = 500
    while True:
        url = (
            f'https://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?'
            f'interval=1&from={from_str}&till={till_str}&start={start}'
        )
        logger.info(f"Запрос минутных данных (start={start}): {url}")
        j = request_moex(session, url, logger)
        if not j or 'candles' not in j or not j['candles'].get('data'):
            logger.error(f"Нет минутных данных для {ticker} на {query_date}")
            break
        data = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
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
        'begin': 'TRADEDATE',
        'open': 'OPEN',
        'close': 'CLOSE',
        'high': 'HIGH',
        'low': 'LOW',
        'volume': 'VOLUME'
    })
    df['SECID'] = ticker
    df = df.dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME'])
    return df[['TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME']].reset_index(drop=True)

# ==== РАБОТА С БД ====
def create_tables(connection: sqlite3.Connection, logger) -> None:
    """
    Создание таблицы 'Futures', если не существует.
    """
    try:
        with connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS Futures (
                            TRADEDATE         TEXT PRIMARY KEY UNIQUE NOT NULL,
                            SECID             TEXT NOT NULL,
                            OPEN              REAL NOT NULL,
                            LOW               REAL NOT NULL,
                            HIGH              REAL NOT NULL,
                            CLOSE             REAL NOT NULL,
                            VOLUME            INTEGER NOT NULL,
                            LSTTRADE          DATE NOT NULL)'''
                           )
        logger.info('Таблица Futures создана в БД.')
    except sqlite3.OperationalError as exception:
        logger.error(f"Ошибка при создании таблицы: {exception}")

def save_to_db(df: pd.DataFrame, connection: sqlite3.Connection, logger) -> None:
    """
    Сохранение обработанного DataFrame в БД.
    """
    if df.empty:
        logger.error("DataFrame пуст, данные не сохранены.")
        return
    try:
        with connection:
            df.to_sql('Futures', connection, if_exists='append', index=False)
        logger.info(f"Сохранено {len(df)} записей в таблицу Futures.")
    except sqlite3.Error as e:
        logger.error(f"Ошибка сохранения данных: {e}")

# ==== ГЛАВНАЯ ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ЗА ПЕРИОД ====
def get_future_date_results(session, start_date: date, end_date: date, ticker: str, connection: sqlite3.Connection, logger) -> None:
    """
    Загрузка и сохранение данных по фьючерсу за период [start_date, end_date].
    """
    query_date = start_date
    while query_date <= end_date:
        date_str = query_date.strftime('%Y-%m-%d')
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM Futures WHERE DATE(TRADEDATE) = ?", (date_str,))
        count = cursor.fetchone()[0]
        if count == 0:
            request_url = (
                f'https://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json?'
                f'date={date_str}&assetcode={ticker}'
            )
            j = request_moex(session, request_url, logger)
            if j is None or 'history' not in j or not j['history'].get('data'):
                logger.info(f"Нет данных по фьючерсам {ticker} за {date_str}")
                query_date += timedelta(days=1)
                continue
            data = [{k: r[i] for i, k in enumerate(j['history']['columns'])} for r in j['history']['data']]
            df = pd.DataFrame(data).dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            if df.empty:
                query_date += timedelta(days=1)
                continue
            df[['SHORTNAME', 'LSTTRADE']] = df.apply(
                lambda x: get_info_future(session, x['SECID'], logger), axis=1, result_type='expand'
            )
            df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"], errors='coerce').dt.date.fillna('2130-01-01')
            df = df[df['LSTTRADE'] > query_date].dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            df = df[df['LSTTRADE'] == df['LSTTRADE'].min()].reset_index(drop=True)
            if df.empty:
                query_date += timedelta(days=1)
                continue
            current_ticker = df.loc[0, 'SECID']
            lasttrade = df.loc[0, 'LSTTRADE']
            minute_df = get_minute_candles(session, current_ticker, query_date, logger)
            minute_df['LSTTRADE'] = lasttrade
            if not minute_df.empty:
                save_to_db(minute_df, connection, logger)
        else:
            # Если данные частично есть, докачать недостающие бары
            cursor.execute("SELECT MAX(TRADEDATE) FROM Futures WHERE DATE(TRADEDATE) = ?", (date_str,))
            max_time_str = cursor.fetchone()[0]
            if max_time_str is None:
                query_date += timedelta(days=1)
                continue
            max_dt = datetime.strptime(max_time_str, '%Y-%m-%d %H:%M:%S')
            threshold_time = time(23, 49, 0)
            if max_dt.time() >= threshold_time:
                logger.info(f"Минутные данные за {date_str} полные, пропускаем.")
                query_date += timedelta(days=1)
                continue
            cursor.execute("SELECT SECID, LSTTRADE FROM Futures WHERE DATE(TRADEDATE) = ? LIMIT 1", (date_str,))
            row = cursor.fetchone()
            if not row:
                query_date += timedelta(days=1)
                continue
            current_ticker = row[0]
            lasttrade = datetime.strptime(str(row[1]), '%Y-%m-%d').date() if isinstance(row[1], str) else row[1]
            from_dt = max_dt + timedelta(minutes=1)
            from_str = from_dt.isoformat()
            till_dt = datetime.combine(query_date, time(23, 59, 59))
            till_str = till_dt.isoformat()
            minute_df = get_minute_candles(session, current_ticker, query_date, logger, from_str, till_str)
            minute_df['LSTTRADE'] = lasttrade
            if not minute_df.empty:
                save_to_db(minute_df, connection, logger)
        query_date += timedelta(days=1)
        cursor.close()

# ==== ФУНКЦИЯ ДЛЯ ГОДОВОЙ БД ====
def main_for_year(ticker: str, year: int, output_dir: Path, provider: str, start_date: date, end_date: date) -> None:
    """
    Загружает минутные данные за один календарный год в отдельную SQLite базу.
    """
    db_name = f"minutes_{ticker}_{year}.sqlite"
    db_path = output_dir / db_name
    log_file = output_dir / 'log' / f'{ticker.lower()}_{provider}_{year}_download.txt'
    logger = init_logger(log_file)
    logger.info(f'Старт обработки минутных данных {ticker} за {year}')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(str(db_path))
        create_tables(connection, logger)
        with requests.Session() as session:
            get_future_date_results(
                session,
                start_date,
                end_date,
                ticker,
                connection,
                logger
            )
    except Exception as e:
        logger.error(f"Ошибка main_for_year: {e}")
    finally:
        try:
            connection.execute("VACUUM")
            logger.info("VACUUM выполнен.")
        except Exception as e:
            logger.error(f"Ошибка при VACUUM: {e}")
        connection.close()
        logger.info(f'Завершена загрузка за {year}, БД: {db_path}')

# ==== ЗАПУСК ПО ВСЕМ ГОДАМ ====
if __name__ == "__main__":
    first_year = start_date_global.year
    last_year = datetime.now().year
    for year in range(first_year, last_year + 1):
        start_of_year = date(year, 1, 1)
        end_of_year = date(year, 12, 31) if year != last_year else datetime.now().date()
        main_for_year(ticker, year, output_dir, provider, start_of_year, end_of_year)
