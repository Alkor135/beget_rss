"""
Скрипт скачивает минутные данные из MOEX ISS API и сохраняет их в базу данных SQLite.
"""

from pathlib import Path
import sqlite3
from datetime import datetime, timedelta, date
import requests
import pandas as pd


def request_moex(session, url, retries = 3, timeout = 5):
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
                            TRADEDATE         DATE PRIMARY KEY UNIQUE NOT NULL,
                            SECID             TEXT NOT NULL,
                            OPEN              REAL NOT NULL,
                            LOW               REAL NOT NULL,
                            HIGH              REAL NOT NULL,
                            CLOSE             REAL NOT NULL,
                            VOLUME            INTEGER NOT NULL,
                            LSTTRADE          DATE NOT NULL)'''
                           )
        print('Taблицы в БД созданы')
    except sqlite3.OperationalError as exception:
        print(f"Ошибка при создании БД: {exception}")

def get_info_future(session, security):
    """Запрашивает у MOEX информацию по инструменту"""
    # print(security)
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

def get_minute_candles(session, ticker: str, start_date: date) -> pd.DataFrame:
    """Получает все минутные данные по фьючерсу за указанную дату с учетом пагинации"""
    all_data = []
    start = 0
    page_size = 500  # MOEX ISS API возвращает до 500 записей за запрос

    while True:
        url = (
            f'https://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?'
            f'interval=1&from={start_date.strftime("%Y-%m-%d")}&till={start_date.strftime("%Y-%m-%d")}'
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

        # Если получено меньше записей, чем page_size, это последняя страница
        if len(data) < page_size:
            break

    if not all_data:
        print(f"Нет данных для {ticker} на {start_date}")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Переименовываем столбцы для соответствия таблице Futures
    df = df.rename(columns={
        'begin': 'TRADEDATE',
        'open': 'OPEN',
        'close': 'CLOSE',
        'high': 'HIGH',
        'low': 'LOW',
        'volume': 'VOLUME'
    })

    # Добавляем TRADEDATE и TRADETIME
    df['SECID'] = ticker

    # # Удаляем лишние столбцы
    # required_columns = ['TRADEDATE', 'TRADETIME', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME', 'LSTTRADE']
    # df = df.drop(columns=[col for col in df.columns if col not in required_columns], errors='ignore')

    # Удаляем строки с NaN в критических полях
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
    today_date = datetime.now().date()  # Текущая дата и время
    while start_date <= today_date:  # Пока start_date меньше текущей даты
        # Проверяем наличие даты в поле TRADEDATE
        query = "SELECT 1 FROM Futures WHERE TRADEDATE = ? LIMIT 1"
        cursor.execute(query, (start_date.strftime('%Y-%m-%d'),))
        result = cursor.fetchone()

        # Нет записей с такой датой в БД, запрашиваем данные с MOEX ISS API
        if result is None:
            request_url = (  # Формируем URL запроса к MOEX ISS API всех фьючерсов на дату start_date
                f'https://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json?'
                f'date={start_date.strftime("%Y-%m-%d")}&assetcode={ticker}'
            )
            # print(f'{request_url=}')  # Отладочная информация

            j = request_moex(session, request_url)
            if not j or 'history' not in j or not j['history'].get('data'):
                print(f"Нет данных для {start_date}")
                start_date += timedelta(days=1)
                continue
            # Данные по всем фьючерсам на дату start_date получены
            data = [{k: r[i] for i, k in enumerate(j['history']['columns'])} for r in
                    j['history']['data']]
            df = pd.DataFrame(data)  # Преобразуем данные в DataFrame
            # Очистка данных: удаляем ненужные строки с NaN
            df = pd.DataFrame(data).dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            if len(df) == 0:
                start_date += timedelta(days=1)
                continue

            df[['SHORTNAME', 'LSTTRADE']] = df.apply(
                lambda x: get_info_future(session, x['SECID']), axis=1, result_type='expand'
            )
            df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"], errors='coerce').dt.date.fillna(
                '2130-01-01')
            df = df[df['LSTTRADE'] > start_date].dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            df = df[df['LSTTRADE'] == df['LSTTRADE'].min()].reset_index(drop=True)
            df = df.drop(columns=[
                'OPENPOSITIONVALUE', 'VALUE', 'SETTLEPRICE', 'SWAPRATE', 'WAPRICE',
                'SETTLEPRICEDAY', 'NUMTRADES', 'SHORTNAME', 'CHANGE', 'QTY'
            ], errors='ignore')
            # print(df.to_string(max_rows=20, max_cols=30), '\n')

            current_ticker = df.loc[0, 'SECID']  # Получаем текущий тикер из DataFrame
            lasttrade = df.loc[0, 'LSTTRADE']  # Получаем дату последней торговли

            # Получаем минутные данные для текущего тикера
            minute_df = get_minute_candles(session, current_ticker, start_date)
            minute_df['LSTTRADE'] = lasttrade  # Добавляем дату последней торговли
            # Если минутные данные не пустые, сохраняем их в БД
            if not minute_df.empty:
                save_to_db(minute_df, connection, cursor)

        start_date += timedelta(days=1)

def main(ticker: str, path_db: Path, start_date: date) -> None:
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
            # Находим максимальную дату (без времени)
            cursor.execute("SELECT MAX(DATE(TRADEDATE)) FROM Futures")
            max_trade_date = cursor.fetchone()[0]
            if max_trade_date:
                # Удаляем все записи с максимальной датой
                cursor.execute("DELETE FROM Futures WHERE DATE(TRADEDATE) = ?",
                               (max_trade_date,))
                connection.commit()
                print(f"Удалены записи с датой: {max_trade_date}")

            # Меняем стартовую дату на удаленную дату
            start_date = datetime.strptime(max_trade_date, "%Y-%m-%d").date()
            print(f"Начальная дата для загрузки данных: {start_date}")

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
        print("Соединение с базой данных закрыто.")


if __name__ == '__main__':
    # Настройка базы данных
    ticker: str = 'RTS'  # Тикер фьючерса
    # Путь к базе данных с минутными барами фьючерсов
    path_db: Path = Path(rf'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_minute_2025.db')
    # Начальная дата для загрузки данных
    start_date: date = datetime.strptime('2025-06-02', "%Y-%m-%d").date()

    main(ticker, path_db, start_date)