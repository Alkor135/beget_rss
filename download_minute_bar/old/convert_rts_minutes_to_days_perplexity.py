"""
Perplexity
Конвертирует минутные котировки по файлам баз данных за года в дневные котировки.
Для минутных данных используется набор файлов SQLite — по одному на год.
Формируются дневные свечи с 21:00 предыдущей сессии до 20:59:59 текущей.
Обрабатываются rollovers — коррекция разрывов между контрактами.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, time
import logging
import os
import yaml

# --- Чтение настроек ---
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings['ticker']
ticker_lc = ticker.lower()
provider = settings['provider']
output_dir = Path(settings['output_dir']).resolve()
time_start = settings['time_start']  # например "21:00:00"
time_end = settings['time_end']      # например "20:59:59"
path_db_day = Path(settings.get('path_db_day').replace('{ticker}', ticker))
minutes_db_dir = Path(settings['path_db_minute']).resolve()  # папка с годовыми БД

# --- Логирование ---
log_file = output_dir / 'log' / f'{ticker_lc}_{provider}_convert_minutes_to_days.txt'
log_file.parent.mkdir(exist_ok=True, parents=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Файл
fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)
# Консоль
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# --- Функции для работы с дневными котировками (без изменений) ---

def create_tables(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS Futures (
                TRADEDATE DATE PRIMARY KEY UNIQUE NOT NULL,
                OPEN REAL NOT NULL,
                LOW REAL NOT NULL,
                HIGH REAL NOT NULL,
                CLOSE REAL NOT NULL,
                SECID TEXT NOT NULL,
                LSTTRADE TEXT NOT NULL
            )
        ''')
    logger.info("Таблица Futures в дневной БД готова.")

def get_sorted_dates(cursor) -> list:
    cursor.execute("SELECT DISTINCT DATE(TRADEDATE) AS trade_date FROM Futures ORDER BY trade_date DESC")
    return [row[0] for row in cursor.fetchall()]

def get_daily_candle(cursor, start: str, end: str):
    # Код из вашего get_daily_candle без изменений, вставьте сюда ваш оригинальный код
    # ...
    # Для краткости здесь не вставляю полностью — предположим, что функция используется как из вашего скрипта.

    # Чтобы не повторяться - вставляю полную функцию ниже отдельно
    pass

def save_daily_candle(connection: sqlite3.Connection, cursor, candle: tuple) -> None:
    # Сохраняем как в вашем скрипте (без изменений)
    pass

def delete_latest_record(connection: sqlite3.Connection, cursor) -> None:
    # Удаляем последнюю запись, если нужно (без изменений)
    pass

# --- Новая логика для работы с годовыми минутными базами ---

def iterate_minutes_dbs(minute_db_dir: Path, ticker: str):
    """
    Возвращает генератор кортежей:
    (year, Path к .sqlite файлу минутных данных)
    Файлы ожидаются с названием формата minutes_{ticker}_{year}.sqlite
    """
    for file_path in minute_db_dir.glob(f"minutes_{ticker}_*.sqlite"):
        try:
            # Извлекаем год из имени файла, ожидаем minutes_RTS_2014.sqlite
            year_str = file_path.stem.split('_')[-1]
            year = int(year_str)
            yield year, file_path
        except Exception as e:
            logger.warning(f"Пропускаем файл {file_path} с ошибкой в имени: {e}")

def main():
    # Подключаемся к дневной базе
    print(f"Подключение к БД: {path_db_day}")

    os.makedirs(os.path.dirname(path_db_day), exist_ok=True)
    connection_day = sqlite3.connect(str(path_db_day))
    cursor_day = connection_day.cursor()
    create_tables(connection_day)

    # Удаляем последнюю запись, если нужна очистка
    delete_latest_record(connection_day, cursor_day)

    # Получаем все минутные базы
    minutes_dbs = list(iterate_minutes_dbs(minutes_db_dir, ticker))
    if not minutes_dbs:
        logger.error("Не найдены базы с минутными данными по паттерну в папке.")
        return

    # Для каждого годового файла
    for year, db_path in sorted(minutes_dbs):
        logger.info(f"Обработка минутной БД за {year}: {db_path}")
        connection_min = sqlite3.connect(str(db_path))
        cursor_min = connection_min.cursor()

        try:
            dates = get_sorted_dates(cursor_min)
            logger.info(f"Всего дат в минутных данных за {year}: {len(dates)}")

            for date_end, date_start in zip(dates, dates[1:] + ['1970-01-01']):
                start = f"{date_start} {time_start}"
                end = f"{date_end} {time_end}"

                candle = get_daily_candle(cursor_min, start, end)
                if candle:
                    save_daily_candle(connection_day, cursor_day, candle)
                else:
                    logger.info(f"Нет данных для периода {start} - {end}")

        finally:
            cursor_min.close()
            connection_min.close()

    # Закрываем дневную базу
    cursor_day.close()
    connection_day.close()
    logger.info("Конвертация минутных данных в дневные завершена.")

if __name__ == "__main__":
    # Полная версия get_daily_candle здесь, чтобы скрипт был самодостаточным
    # Вставляем из вашего кода функцию get_daily_candle (полный текст)

    def get_daily_candle(cursor, start: str, end: str) -> tuple:
        trade_date = datetime.strptime(end, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

        query_count = """
            SELECT COUNT(DISTINCT SECID) FROM Futures WHERE TRADEDATE BETWEEN ? AND ?
        """
        cursor.execute(query_count, (start, end))
        num_secid = cursor.fetchone()[0]

        if num_secid == 0:
            return None

        query_last = """
            SELECT SECID, LSTTRADE, CLOSE FROM Futures WHERE TRADEDATE = (
                SELECT MAX(TRADEDATE) FROM Futures WHERE TRADEDATE BETWEEN ? AND ?
            ) LIMIT 1
        """
        cursor.execute(query_last, (start, end))
        last_result = cursor.fetchone()
        if not last_result:
            return None
        secid, lsttrade, close = last_result

        if num_secid == 1:
            query = """
                SELECT 
                    (SELECT OPEN FROM Futures WHERE TRADEDATE = (
                        SELECT MIN(TRADEDATE) FROM Futures WHERE TRADEDATE BETWEEN ? AND ?
                    ) LIMIT 1) AS OPEN,
                    MIN(LOW) AS LOW,
                    MAX(HIGH) AS HIGH
                FROM Futures
                WHERE TRADEDATE BETWEEN ? AND ?
            """
            cursor.execute(query, (start, end, start, end))
            result = cursor.fetchone()
            if result and all(result[i] is not None for i in range(3)):
                return (trade_date, result[0], result[1], result[2], close, secid, lsttrade)
            return None

        else:
            query_new_start = """
                SELECT MIN(TRADEDATE) FROM Futures 
                WHERE SECID = ? AND TRADEDATE BETWEEN ? AND ?
            """
            cursor.execute(query_new_start, (secid, start, end))
            new_start = cursor.fetchone()[0]
            if not new_start:
                return None

            query_old_end = """
                SELECT MAX(TRADEDATE) FROM Futures 
                WHERE TRADEDATE < ? AND TRADEDATE BETWEEN ? AND ?
            """
            cursor.execute(query_old_end, (new_start, start, end))
            old_end = cursor.fetchone()[0]
            if not old_end:
                query_new = """
                    SELECT 
                        (SELECT OPEN FROM Futures WHERE TRADEDATE = ? LIMIT 1) AS OPEN,
                        MIN(LOW) AS LOW,
                        MAX(HIGH) AS HIGH
                    FROM Futures
                    WHERE TRADEDATE >= ? AND TRADEDATE <= ?
                """
                cursor.execute(query_new, (new_start, new_start, end))
                result_new = cursor.fetchone()
                if result_new and all(result_new[i] is not None for i in range(3)):
                    return (trade_date, result_new[0], result_new[1], result_new[2], close, secid, lsttrade)
                return None

            query_close_old = "SELECT CLOSE FROM Futures WHERE TRADEDATE = ? LIMIT 1"
            cursor.execute(query_close_old, (old_end,))
            last_close_old = cursor.fetchone()[0]

            query_open_new = "SELECT OPEN FROM Futures WHERE TRADEDATE = ? LIMIT 1"
            cursor.execute(query_open_new, (new_start,))
            first_open_new = cursor.fetchone()[0]

            gap = first_open_new - last_close_old

            query_old = """
                SELECT 
                    (SELECT OPEN FROM Futures WHERE TRADEDATE = (
                        SELECT MIN(TRADEDATE) FROM Futures WHERE TRADEDATE BETWEEN ? AND ?
                    ) LIMIT 1) + ? AS ADJ_OPEN,
                    MIN(LOW) + ? AS ADJ_LOW,
                    MAX(HIGH) + ? AS ADJ_HIGH
                FROM Futures
                WHERE TRADEDATE BETWEEN ? AND ?
            """
            old_start = start
            old_end_str = old_end
            cursor.execute(query_old, (old_start, old_end_str, gap, gap, gap, old_start, old_end_str))
            result_old = cursor.fetchone()
            if not result_old or any(result_old[i] is None for i in range(3)):
                query_new_only = """
                    SELECT 
                        (SELECT OPEN FROM Futures WHERE TRADEDATE = ? LIMIT 1) AS OPEN,
                        MIN(LOW) AS LOW,
                        MAX(HIGH) AS HIGH
                    FROM Futures
                    WHERE TRADEDATE >= ? AND TRADEDATE <= ?
                """
                cursor.execute(query_new_only, (new_start, new_start, end))
                result_new_only = cursor.fetchone()
                if result_new_only and all(result_new_only[i] is not None for i in range(3)):
                    return (trade_date, result_new_only[0], result_new_only[1], result_new_only[2], close, secid, lsttrade)
                return None

            adj_open, adj_low_old, adj_high_old = result_old

            query_new = """
                SELECT 
                    MIN(LOW) AS LOW,
                    MAX(HIGH) AS HIGH
                FROM Futures
                WHERE TRADEDATE BETWEEN ? AND ?
            """
            cursor.execute(query_new, (new_start, end))
            result_new = cursor.fetchone()
            if not result_new or any(result_new[i] is None for i in range(2)):
                return None
            low_new, high_new = result_new

            overall_low = min(adj_low_old, low_new)
            overall_high = max(adj_high_old, high_new)

            return (trade_date, adj_open, overall_low, overall_high, close, secid, lsttrade)

    # Остальной код main без изменений
    # Далее вставляем нужные вспомогательные функции save_daily_candle, delete_latest_record из вашего скрипта
    # Для полноты:

    def save_daily_candle(connection: sqlite3.Connection, cursor, candle: tuple) -> None:
        if candle:
            query_check = "SELECT COUNT(*) FROM Futures WHERE TRADEDATE = ?"
            cursor.execute(query_check, (candle[0],))
            if cursor.fetchone()[0] > 0:
                logger.info(f"Запись для {candle[0]} уже существует, пропускаем.")
                return

            query = """
                INSERT INTO Futures (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            with connection:
                try:
                    cursor.execute(query, candle)
                    connection.commit()
                    logger.info(f"Сохранена дневная свечка для {candle[0]}")
                except sqlite3.Error as e:
                    logger.error(f"Ошибка при сохранении дневной свечки: {e}")

    def delete_latest_record(connection: sqlite3.Connection, cursor) -> None:
        query_max = "SELECT MAX(TRADEDATE) FROM Futures"
        cursor.execute(query_max)
        max_date = cursor.fetchone()[0]
        if max_date:
            query_delete = "DELETE FROM Futures WHERE TRADEDATE = ?"
            cursor.execute(query_delete, (max_date,))
            connection.commit()
            logger.info(f"Удалена последняя запись для даты {max_date}")
        else:
            logger.info("Таблица Futures пуста, удалять нечего.")

    main()  # Запуск основного процесса
