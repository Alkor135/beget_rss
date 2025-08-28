"""
Скрипт скачивает минутные данные из MOEX ISS API и сохраняет их в базу данных SQLite.
"""

from pathlib import Path
import sqlite3
from datetime import datetime, timedelta, date, time
import requests
import pandas as pd


# Параметры
ticker: str = 'RTS'  # Тикер фьючерса
# Путь к базе данных с минутными барами фьючерсов
path_db: Path = Path(rf'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_minute_2025.db')
# Начальная дата для загрузки данных
start_date: date = datetime.strptime('2025-06-02', "%Y-%m-%d").date()


def request_moex(session, url, retries = 5, timeout = 10):
    """Функция запроса данных с повторными попытками"""
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка запроса {url} (попытка {attempt + 1}): {e}")
            if attempt == retries - 1:
                return None

def create_tables(connection: sqlite3.Connection) -> None:
    """ Функция создания таблиц в БД если их нет"""
    try:
        with connection:
            connection.execute('''CREATE TABLE if not exists Futures (
                            TRADEDATE         TEXT PRIMARY KEY UNIQUE NOT NULL,
                            SECID             TEXT NOT NULL,
                            OPEN              REAL NOT NULL,
                            LOW               REAL NOT NULL,
                            HIGH              REAL NOT NULL,
                            CLOSE             REAL NOT NULL,
                            VOLUME            INTEGER NOT NULL,
                            LSTTRADE          DATE NOT NULL)'''
                           )
        print('Таблицы в БД созданы')
    except sqlite3.OperationalError as exception:
        print(f"Ошибка при создании БД: {exception}")

def get_info_future(session, security):
    """Запрашивает у MOEX информацию по инструменту"""
    url = f'https://iss.moex.com/iss/securities/{security}.json'
    j = request_moex(session, url)

    if not j:
        return pd.Series(["", "2130-01-01"])  # Гарантируем, что всегда 2 значения

    data = [{k: r[i] for i, k in enumerate(j['description']['columns'])} for r in j['description']['data']]
    df = pd.DataFrame(data)

    shortname = df.loc[df['name'] == 'SHORTNAME', 'value'].values[0] if 'SHORTNAME' in df['name'].values else ""
    lsttrade = df.loc[df['name'] == 'LSTTRADE', 'value'].values[0] if 'LSTTRADE' in df['name'].values else \
               df.loc[df['name'] == 'LSTDELDATE', 'value'].values[0] if 'LSTDELDATE' in df['name'].values else "2130-01-01"

    return pd.Series([shortname, lsttrade])  # Гарантируем возврат 2 значений

def get_minute_candles(session, ticker: str, start_date: date, from_str: str = None, till_str: str = None) -> pd.DataFrame:
    """Получает все минутные данные по фьючерсу за указанную дату с учетом пагинации"""
    if from_str is None:
        from_str = datetime.combine(start_date, time(0, 0)).isoformat()
    if till_str is None:
        till_str = datetime.combine(start_date, time(23, 59, 59)).isoformat()

    all_data = []
    start = 0
    page_size = 500  # MOEX ISS API возвращает до 500 записей за запрос

    while True:
        url = (
            f'https://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?'
            f'interval=1&from={from_str}&till={till_str}'
            f'&start={start}'
        )
        print(f"Запрос минутных данных (start={start}): {url}")

        j = request_moex(session, url)
        if not j or 'candles' not in j or not j['candles'].get('data'):
            print(f"Нет минутных данных для {ticker} на {start_date}")
            break

        data = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
        if not data:
            break

        all_data.extend(data)
        start += page_size

        if len(data) < page_size:
            break

    if not all_data:
        print(f"Нет данных для {ticker} на {start_date}")
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
    print(df.to_string(max_rows=6, max_cols=18), '\n')

    return df[['TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME']].reset_index(drop=True)


def save_to_db(df: pd.DataFrame, connection: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Сохраняет DataFrame в таблицу Futures"""
    if df.empty:
        print("DataFrame пуст, данные не сохранены")
        return

    try:
        with connection:
            df.to_sql('Futures', connection, if_exists='append', index=False)
        print(f"Сохранено {len(df)} записей в таблицу Futures")
    except sqlite3.Error as e:
        print(f"Ошибка при сохранении данных в БД: {e}")

def get_future_date_results(
        session,
        start_date: date,
        ticker: str,
        connection: sqlite3.Connection,
        cursor: sqlite3.Cursor) -> None:
    """Получает данные по фьючерсам с MOEX ISS API и сохраняет их в базу данных."""
    today_date = datetime.now().date()  # Текущая дата
    while start_date <= today_date:
        date_str = start_date.strftime('%Y-%m-%d')
        # Проверяем количество записей в БД за дату
        cursor.execute("SELECT COUNT(*) FROM Futures WHERE DATE(TRADEDATE) = ?", (date_str,))
        count = cursor.fetchone()[0]

        if count == 0:
            # Нет минутных данных в БД, запрашиваем данные о торгуемых фьючерсах на дату
            # За текущую дату торгуемые тикеры доступны после 19:05, после окончания основной сессии
            request_url = (
                f'https://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json?'
                f'date={date_str}&assetcode={ticker}'
            )

            j = request_moex(session, request_url)
            if j is None:
                print(f"Ошибка получения данных для {start_date}. Прерываем процесс, чтобы повторить попытку в следующий запуск.")
                break
            elif 'history' not in j or not j['history'].get('data'):
                print(f"Нет данных по торгуемым фьючерсам {ticker} за {start_date}")
                start_date += timedelta(days=1)
                continue

            data = [{k: r[i] for i, k in enumerate(j['history']['columns'])} for r in j['history']['data']]
            df = pd.DataFrame(data).dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            if len(df) == 0:
                start_date += timedelta(days=1)
                continue

            df[['SHORTNAME', 'LSTTRADE']] = df.apply(
                lambda x: get_info_future(session, x['SECID']), axis=1, result_type='expand'
            )
            df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"], errors='coerce').dt.date.fillna('2130-01-01')
            df = df[df['LSTTRADE'] > start_date].dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            df = df[df['LSTTRADE'] == df['LSTTRADE'].min()].reset_index(drop=True)
            df = df.drop(columns=[
                'OPENPOSITIONVALUE', 'VALUE', 'SETTLEPRICE', 'SWAPRATE', 'WAPRICE',
                'SETTLEPRICEDAY', 'NUMTRADES', 'SHORTNAME', 'CHANGE', 'QTY'
            ], errors='ignore')

            current_ticker = df.loc[0, 'SECID']
            lasttrade = df.loc[0, 'LSTTRADE']

            # Получаем минутные данные
            minute_df = get_minute_candles(session, current_ticker, start_date)
            minute_df['LSTTRADE'] = lasttrade
            if not minute_df.empty:
                save_to_db(minute_df, connection, cursor)

        else:
            # Есть минутные данные за дату, проверяем полноту
            cursor.execute("SELECT MAX(TRADEDATE) FROM Futures WHERE DATE(TRADEDATE) = ?", (date_str,))
            max_time_str = cursor.fetchone()[0]
            max_dt = datetime.strptime(max_time_str, '%Y-%m-%d %H:%M:%S')

            threshold_time = time(23, 49, 0)
            is_today = start_date == today_date

            if not is_today and max_dt.time() >= threshold_time:
                print(f"Минутные данные за {start_date} полные, пропускаем дату {start_date}")
                start_date += timedelta(days=1)
                continue

            # Неполные минутные данные или сегодняшний день (после 19:05), докачиваем
            cursor.execute("SELECT SECID, LSTTRADE FROM Futures WHERE DATE(TRADEDATE) = ? LIMIT 1", (date_str,))
            row = cursor.fetchone()
            current_ticker = row[0]
            lasttrade = datetime.strptime(row[1], '%Y-%m-%d').date() if isinstance(row[1], str) else row[1]

            from_dt = max_dt + timedelta(minutes=1)
            from_str = from_dt.isoformat()

            if is_today:
                till_dt = datetime.now()
            else:
                till_dt = datetime.combine(start_date, time(23, 59, 59))
            till_str = till_dt.isoformat()

            minute_df = get_minute_candles(session, current_ticker, start_date, from_str, till_str)
            minute_df['LSTTRADE'] = lasttrade
            if not minute_df.empty:
                save_to_db(minute_df, connection, cursor)

        start_date += timedelta(days=1)

def main(
        ticker: str = ticker,
        path_db: Path = path_db,
        start_date: date = start_date) -> None:
    """
    Основная функция: подключается к базе данных, создает таблицы и загружает данные по фьючерсам.
    """
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
            # Находим максимальную дату
            cursor.execute("SELECT MAX(DATE(TRADEDATE)) FROM Futures")
            max_trade_date = cursor.fetchone()[0]
            if max_trade_date:
                # Устанавливаем start_date на максимальную дату для проверки полноты
                start_date = datetime.strptime(max_trade_date, "%Y-%m-%d").date()
                print(f"Начальная дата для загрузки минутных данных: {start_date}")

        with requests.Session() as session:
            get_future_date_results(session, start_date, ticker, connection, cursor)

    except Exception as e:
        print(f"Ошибка в main: {e}")

    finally:
        # Выполняем команду VACUUM
        cursor.execute("VACUUM")
        print("VACUUM выполнен: база данных оптимизирована")

        # Закрываем курсор и соединение
        cursor.close()
        connection.close()
        print(f"Соединение с минутной БД {path_db} по фьючерсам {ticker} закрыто.")


if __name__ == '__main__':
    main(ticker, path_db, start_date)