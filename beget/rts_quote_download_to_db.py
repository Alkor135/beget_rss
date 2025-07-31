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
from logging.handlers import TimedRotatingFileHandler

# Настройка логирования с ротацией по времени
log_handler = TimedRotatingFileHandler(
    '/home/user/rss_scraper/log/rts_quote_download_to_db.log',
    when='midnight',  # Новый файл каждый день в полночь
    interval=1,
    backupCount=2  # Хранить логи за 2 дней
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(log_handler)

def create_tables(connection: sqlite3.Connection) -> None:
    """ Функция создания таблицы в БД если её нет"""
    with connection:
        try:
            connection.execute('''CREATE TABLE if not exists Futures (
                            TRADEDATE         DATE PRIMARY KEY UNIQUE NOT NULL,
                            SECID             TEXT NOT NULL,
                            OPEN              REAL NOT NULL,
                            LOW               REAL NOT NULL,
                            HIGH              REAL NOT NULL,
                            CLOSE             REAL NOT NULL,
                            LSTTRADE          DATE NOT NULL)'''
                           )
            connection.commit()
            logging.info('Таблица Futures в БД создана.')
        except sqlite3.OperationalError as e:
            logging.error(f"Ошибка при создании таблицы Futures: {e}")

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
        return pd.Series(["", "2130-01-01"])  # Гарантируем, что всегда 2 значения

    data = [{k: r[i] for i, k in enumerate(j['description']['columns'])} for r in
            j['description']['data']]
    df = pd.DataFrame(data)

    shortname = df.loc[df['name'] == 'SHORTNAME', 'value'].values[0] if 'SHORTNAME' in df[
        'name'].values else ""
    lsttrade = df.loc[df['name'] == 'LSTTRADE', 'value'].values[0] if 'LSTTRADE' in df[
        'name'].values else \
        df.loc[df['name'] == 'LSTDELDATE', 'value'].values[0] if 'LSTDELDATE' in df[
            'name'].values else "2130-01-01"

    return pd.Series([shortname, lsttrade])  # Гарантируем возврат 2 значений

def get_future_date_results(
        session: requests.Session,
        tradedate: datetime.date,
        ticker: str,
        connection: sqlite3.Connection,
        cursor: sqlite3.Cursor
        ) -> None:
    """
    Получает данные по фьючерсам с MOEX ISS API и сохраняет их в базу данных.

    :param session: Сессия requests для выполнения HTTP-запросов.
    :param tradedate: Дата начала загрузки данных.
    :param ticker: Тикер инструмента (например, 'RTS').
    :param connection: Соединение с базой данных SQLite.
    :param cursor: Курсор для выполнения SQL-запросов.
    """
    today_date = datetime.now().date()  # Текущая дата и время
    while tradedate < today_date:
        # Проверяем наличие даты в поле TRADEDATE
        query = "SELECT 1 FROM Futures WHERE TRADEDATE = ? LIMIT 1"
        cursor.execute(query, (tradedate.strftime('%Y-%m-%d'),))
        result = cursor.fetchone()

        # Нет записи с такой датой
        if result is None:
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

            data = [{k: r[i] for i, k in enumerate(j['history']['columns'])} for r in
                    j['history']['data']]
            df = pd.DataFrame(data).dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            # print(df.to_string(max_rows=20, max_cols=20))

            if len(df) == 0:
                tradedate += timedelta(days=1)
                continue

            df[['SHORTNAME', 'LSTTRADE']] = df.apply(
                lambda x: get_info_future(session, x['SECID']), axis=1, result_type='expand'
            )
            df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"], errors='coerce').dt.date.fillna(
                '2130-01-01')
            df = df[df['LSTTRADE'] > tradedate].dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            df = df[df['LSTTRADE'] == df['LSTTRADE'].min()].reset_index(drop=True)

            if len(df) == 1 and not df['OPEN'].isnull().values.any():
                # Добавляем строку в таблицу Futures
                cursor.execute(
                    "INSERT INTO Futures (TRADEDATE, SECID, OPEN, LOW, HIGH, CLOSE, LSTTRADE) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (str(df.loc[0]['TRADEDATE']), str(df.loc[0]['SECID']),
                     float(df.loc[0]['OPEN']), float(df.loc[0]['LOW']),
                     float(df.loc[0]['HIGH']), float(df.loc[0]['CLOSE']),
                     str(df.loc[0]['LSTTRADE']))
                )
                connection.commit()

                df = df.drop([
                    'OPENPOSITIONVALUE', 'VALUE', 'SETTLEPRICE', 'SWAPRATE', 'WAPRICE',
                    'SETTLEPRICEDAY', 'NUMTRADES', 'SHORTNAME', 'CHANGE', 'QTY'
                ], axis=1)
                logging.info(f"Данные для {tradedate}: {df.to_string(max_rows=5, max_cols=20)}")
                logging.info('Строка записана в БД')
            else:
                logging.warning(f"Данные для {tradedate} не соответствуют условиям записи")
        tradedate += timedelta(days=1)

def main(ticker, path_db, start_date):
    """Основная функция"""
    try:
        # Создание директории под БД, если не существует
        path_db.parent.mkdir(parents=True, exist_ok=True)

        # Подключение к базе данных
        connection = sqlite3.connect(str(path_db), check_same_thread=True)
        cursor = connection.cursor()

        # Проверяем наличие таблицы Futures
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Futures'")
        exist_table = cursor.fetchone()
        # Если таблица Futures не существует, создаем её
        if exist_table is None:
            create_tables(connection)

        # Проверяем, есть ли записи в таблице Futures
        cursor.execute("SELECT EXISTS (SELECT 1 FROM Futures) as has_rows")
        exists_rows = cursor.fetchone()[0]
        # Если таблица Futures не пустая
        if exists_rows:
            # Удаляем последнюю запись из БД
            cursor.execute("SELECT MAX(TRADEDATE) FROM Futures")
            max_trade_date = cursor.fetchone()[0]
            if max_trade_date:
                cursor.execute("DELETE FROM Futures WHERE TRADEDATE = ?", (max_trade_date,))
                connection.commit()
                logging.info(f"Удалена последняя запись с датой: {max_trade_date}")

            # Меняем стартовую дату на удаленную дату
            start_date = datetime.strptime(max_trade_date, "%Y-%m-%d").date()

        with requests.Session() as session:
            get_future_date_results(session, start_date, ticker, connection, cursor)

    except Exception as e:
        logging.error(f"Ошибка в main: {e}")

    finally:
        # Выполняем команду VACUUM
        cursor.execute("VACUUM")
        logging.info("VACUUM выполнен: база данных оптимизирована")

        # Закрываем курсор и соединение
        cursor.close()
        connection.close()
        logging.info("Соединение с базой данных закрыто.\n")

if __name__ == '__main__':
    ticker = 'RTS'
    path_db = Path(fr'/home/user/rss_scraper/db_data/{ticker}_day_rss_2025.db')
    start_date = datetime.strptime('2025-01-01', "%Y-%m-%d").date()
    main(ticker, path_db, start_date)