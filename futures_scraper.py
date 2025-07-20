"""
Получение исторических данных по фьючерсам RTS с MOEX ISS API и занесение записей в БД.
Загружать от 2025-01-01
Адаптированный скрипт для Beget
"""
from pathlib import Path
import requests
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import logging
# from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler

# Регистрация адаптера и конвертера для дат
sqlite3.register_adapter(datetime, lambda d: d.isoformat())
sqlite3.register_converter("DATETIME", lambda x: datetime.fromisoformat(x.decode()))

# # Настройка логирования с ротацией
# log_handler = RotatingFileHandler(
#     '/home/user/rss_scraper/futures_scraper.log',
#     maxBytes=5*1024*1024,
#     backupCount=3
# )
# Настройка логирования с ротацией по времени
log_handler = TimedRotatingFileHandler(
    '/home/user/rss_scraper/futures_scraper.log',
    when='midnight',  # Новый файл каждый день в полночь
    interval=1,
    backupCount=7  # Хранить логи за 7 дней
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(log_handler)

def create_tables(connection: sqlite3.Connection) -> None:
    """Создание таблицы Futures в БД, если она не существует"""
    try:
        with connection:
            connection.execute('''
                CREATE TABLE IF NOT EXISTS Futures (
                    TRADEDATE DATETIME PRIMARY KEY UNIQUE NOT NULL,
                    SECID TEXT NOT NULL,
                    OPEN REAL NOT NULL,
                    LOW REAL NOT NULL,
                    HIGH REAL NOT NULL,
                    CLOSE REAL NOT NULL,
                    LSTTRADE DATETIME NOT NULL
                )
            ''')
            logging.info('Таблица Futures в БД создана или уже существует')
    except sqlite3.Error as e:
        logging.error(f"Ошибка при создании таблицы Futures: {e}")

def non_empty_table_futures(connection, cursor):
    """Проверяет, есть ли записи в таблице Futures"""
    with connection:
        return cursor.execute("SELECT 1 FROM Futures LIMIT 1").fetchone() is not None

def tradedate_futures_exists(connection, cursor, tradedate):
    """Проверяет, есть ли дата в таблице Futures"""
    with connection:
        result = cursor.execute('SELECT * FROM Futures WHERE TRADEDATE = ?', (tradedate,)).fetchall()
        return bool(len(result))

def add_tradedate_future(connection, cursor, tradedate, secid, open, low, high, close, lsttrade):
    """Добавляет строку в таблицу Futures"""
    try:
        with connection:
            cursor.execute(
                "INSERT INTO Futures (TRADEDATE, SECID, OPEN, LOW, HIGH, CLOSE, LSTTRADE) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tradedate, secid, open, low, high, close, lsttrade)
            )
            logging.info(f"Добавлена запись для даты {tradedate}")
    except sqlite3.IntegrityError as e:
        logging.error(f"Ошибка вставки данных в таблицу Futures: {e}")

def get_max_date_futures(connection, cursor):
    """Получает максимальную дату из таблицы Futures"""
    with connection:
        result = cursor.execute('SELECT MAX(TRADEDATE) FROM Futures').fetchall()[0][0]
        return result if result else '2014-01-01'

def request_moex(session, url, retries=3, timeout=5):
    """Функция запроса данных с повторными попытками"""
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Ошибка запроса {url} (попытка {attempt + 1}): {e}")
            if attempt == retries - 1:
                return None

def get_info_future(session, security):
    """Запрашивает у MOEX информацию по инструменту"""
    url = f'https://iss.moex.com/iss/securities/{security}.json'
    j = request_moex(session, url)

    if not j:
        return pd.Series(["", "2130-01-01"])  # Гарантируем возврат 2 значений

    data = [{k: r[i] for i, k in enumerate(j['description']['columns'])} for r in j['description']['data']]
    df = pd.DataFrame(data)

    shortname = df.loc[df['name'] == 'SHORTNAME', 'value'].values[0] if 'SHORTNAME' in df['name'].values else ""
    lsttrade = df.loc[df['name'] == 'LSTTRADE', 'value'].values[0] if 'LSTTRADE' in df['name'].values else \
               df.loc[df['name'] == 'LSTDELDATE', 'value'].values[0] if 'LSTDELDATE' in df['name'].values else "2130-01-01"

    return pd.Series([shortname, lsttrade])

def get_future_date_results(session, tradedate, ticker, connection, cursor):
    """Получает данные по фьючерсам с MOEX ISS API и сохраняет их в базу данных"""
    today_date = datetime.now().date()
    while tradedate < today_date:
        if not tradedate_futures_exists(connection, cursor, tradedate):
            url = (
                f'https://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json?'
                f'date={tradedate}&assetcode={ticker}'
            )
            logging.info(f"Запрос данных для {tradedate}: {url}")
            j = request_moex(session, url)
            if not j or 'history' not in j or not j['history'].get('data'):
                logging.warning(f"Нет данных для {tradedate}")
                tradedate += timedelta(days=1)
                continue

            data = [{k: r[i] for i, k in enumerate(j['history']['columns'])} for r in j['history']['data']]
            df = pd.DataFrame(data).dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])

            if df.empty:
                logging.warning(f"Пустой DataFrame для {tradedate}")
                tradedate += timedelta(days=1)
                continue

            df[['SHORTNAME', 'LSTTRADE']] = df.apply(
                lambda x: get_info_future(session, x['SECID']), axis=1, result_type='expand'
            )
            df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"], errors='coerce').dt.date.fillna('2130-01-01')
            df = df[df['LSTTRADE'] > tradedate].dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            df = df[df['LSTTRADE'] == df['LSTTRADE'].min()].reset_index(drop=True)

            if len(df) == 1 and not df['OPEN'].isnull().values.any():
                add_tradedate_future(
                    connection, cursor, df.loc[0]['TRADEDATE'], df.loc[0]['SECID'],
                    float(df.loc[0]['OPEN']), float(df.loc[0]['LOW']),
                    float(df.loc[0]['HIGH']), float(df.loc[0]['CLOSE']),
                    df.loc[0]['LSTTRADE']
                )
                df = df.drop([
                    'OPENPOSITIONVALUE', 'VALUE', 'SETTLEPRICE', 'SWAPRATE', 'WAPRICE',
                    'SETTLEPRICEDAY', 'NUMTRADES', 'SHORTNAME', 'CHANGE', 'QTY'
                ], axis=1, errors='ignore')
                logging.info(f"Данные для {tradedate}:\n{df.to_string(max_rows=5, max_cols=20)}")
            else:
                logging.warning(f"Данные для {tradedate} не соответствуют условиям записи")
        tradedate += timedelta(days=1)

def main():
    """Основная функция"""
    try:
        ticker = 'RTS'
        db_path = Path('/home/user/rss_scraper/RTS_day_rss_2025.db')
        logging.info(f"Запуск сбора данных: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Создание директории, если не существует
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Подключение к базе данных
        connection = sqlite3.connect(str(db_path), check_same_thread=True)
        cursor = connection.cursor()

        # Создание таблицы
        create_tables(connection)

        # Определение стартовой даты
        start_date = datetime.strptime('2025-01-01', "%Y-%m-%d").date()
        if non_empty_table_futures(connection, cursor):
            cursor.execute("SELECT MAX(TRADEDATE) FROM Futures")
            max_trade_date = cursor.fetchone()[0]
            if max_trade_date:
                cursor.execute("DELETE FROM Futures WHERE TRADEDATE = ?", (max_trade_date,))
                connection.commit()
                start_date = datetime.strptime(max_trade_date, "%Y-%m-%d").date() + timedelta(days=1)

        # Запрос данных
        with requests.Session() as session:
            get_future_date_results(session, start_date, ticker, connection, cursor)

        # Выполнение VACUUM
        cursor.execute("VACUUM")
        logging.info("VACUUM выполнен: база данных оптимизирована")

        # Закрытие соединения
        cursor.close()
        connection.close()

    except Exception as e:
        logging.error(f"Ошибка в main: {e}")

if __name__ == '__main__':
    main()