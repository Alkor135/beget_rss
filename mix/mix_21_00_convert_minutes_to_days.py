"""
Конвертирует котировки, минутные бары фьючерса в дневные с заданным временем начала и конца дневной свечи.
Дневная свеча формируется с 21:00:00 предыдущей сессии до 20:59:59 текущей сессии по МСК.
При обнаружении нескольких контрактов в диапазоне (rollover), корректирует цены старого контракта
на разницу (gap) для обеспечения непрерывности.
Удаляет только последнюю запись по дате в поле TRADEDATE, если база существует.
Добавляет только отсутствующие по дате записи, пропуская существующие.
"""

import sqlite3
from pathlib import Path
from datetime import datetime


# Параметры
ticker: str = 'MIX'
# Путь к файлу БД с минутными котировками скаченными с MOEX ISS API
path_db_minutes: Path = Path(rf'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_minute_2025.db')
# Путь к файлу БД с дневными котировками (с 21:00 предыдущей сессии)
path_db_day: Path = Path(rf'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db')
time_start = '21:00:00'  # Время старта поиска минутных баров в предыдущую сессию
time_end = '20:59:59'  # Время окончания поиска минутных баров за текущую сессию

def create_tables(connection: sqlite3.Connection) -> None:
    """Функция создания таблицы в БД, если её нет"""
    with connection:
        try:
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
            connection.commit()
            print('Таблица Futures в БД создана.')
        except sqlite3.OperationalError as e:
            print(f"Ошибка при создании таблицы Futures: {e}")

def get_sorted_dates(connection, cursor, db_path: Path) -> list:
    """Получает список всех уникальных дат из таблицы Futures, отсортированных по убыванию."""
    cursor.execute(
        "SELECT DISTINCT DATE(TRADEDATE) AS trade_date FROM Futures ORDER BY trade_date DESC")
    dates: list = [row[0] for row in cursor.fetchall()]
    return dates

def get_daily_candle(cursor, start: str, end: str) -> tuple:
    """
    Получает дневную свечку из минутных данных за указанный диапазон времени.
    Если в диапазоне несколько контрактов (rollover), корректирует цены старого контракта на gap.

    Args:
        cursor: Курсор SQLite для выполнения запросов.
        start: Начало периода в формате 'YYYY-MM-DD HH:MM:SS'.
        end: Конец периода в формате 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        tuple: (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE) или None, если данных нет.
    """
    # Извлекаем дату из end для TRADEDATE
    trade_date = datetime.strptime(end, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

    # Проверяем количество уникальных SECID в диапазоне
    query_count = """
        SELECT COUNT(DISTINCT SECID) FROM Futures WHERE TRADEDATE BETWEEN ? AND ?
    """
    cursor.execute(query_count, (start, end))
    num_secid = cursor.fetchone()[0]

    if num_secid == 0:
        return None

    # Получаем SECID, LSTTRADE из последнего бара
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
        # Обычный случай: один контракт
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
        # Rollover: несколько контрактов, предполагаем два (старый и новый)
        # Находим начало нового контракта (min TRADEDATE где SECID = last_secid)
        query_new_start = """
            SELECT MIN(TRADEDATE) FROM Futures 
            WHERE SECID = ? AND TRADEDATE BETWEEN ? AND ?
        """
        cursor.execute(query_new_start, (secid, start, end))
        new_start = cursor.fetchone()[0]
        if not new_start:
            return None

        # Последний бар старого контракта (max TRADEDATE < new_start)
        query_old_end = """
            SELECT MAX(TRADEDATE) FROM Futures 
            WHERE TRADEDATE < ? AND TRADEDATE BETWEEN ? AND ?
        """
        cursor.execute(query_old_end, (new_start, start, end))
        old_end = cursor.fetchone()[0]
        if not old_end:
            # Нет старой части, обрабатываем как один контракт (только новый)
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

        # Получаем last_close_old
        query_close_old = """
            SELECT CLOSE FROM Futures WHERE TRADEDATE = ? LIMIT 1
        """
        cursor.execute(query_close_old, (old_end,))
        last_close_old = cursor.fetchone()[0]

        # Получаем first_open_new
        query_open_new = """
            SELECT OPEN FROM Futures WHERE TRADEDATE = ? LIMIT 1
        """
        cursor.execute(query_open_new, (new_start,))
        first_open_new = cursor.fetchone()[0]

        # Вычисляем gap
        gap = first_open_new - last_close_old

        # Данные для старой части (с корректировкой)
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
        old_end_str = old_end  # Уже строка
        cursor.execute(query_old, (old_start, old_end_str, gap, gap, gap, old_start, old_end_str))
        result_old = cursor.fetchone()
        if not result_old or any(result_old[i] is None for i in range(3)):
            # Если нет старой части или данных, обрабатываем только новую
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

        # Данные для новой части
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

        # Аггрегируем
        overall_low = min(adj_low_old, low_new)
        overall_high = max(adj_high_old, high_new)

        return (trade_date, adj_open, overall_low, overall_high, close, secid, lsttrade)

def save_daily_candle(connection: sqlite3.Connection, cursor, candle: tuple) -> None:
    """
    Сохраняет дневную свечку в таблицу Futures, если записи для этой даты нет.

    Args:
        connection: Соединение с базой данных.
        cursor: Курсор SQLite для выполнения запросов.
        candle: Кортеж (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE).
    """
    if candle:
        # Проверяем наличие записи для этой даты
        query_check = """
            SELECT COUNT(*) FROM Futures WHERE TRADEDATE = ?
        """
        cursor.execute(query_check, (candle[0],))
        if cursor.fetchone()[0] > 0:
            print(f"Запись для {candle[0]} уже существует, пропускаем.")
            return

        query = """
            INSERT INTO Futures (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with connection:
            try:
                cursor.execute(query, candle)
                connection.commit()
                print(f"Сохранена дневная свечка для {candle[0]}")
            except sqlite3.Error as e:
                print(f"Ошибка при сохранении дневной свечки: {e}")

def delete_latest_record(connection: sqlite3.Connection, cursor) -> None:
    """
    Удаляет последнюю запись по максимальной дате в поле TRADEDATE.
    """
    query_max = """
        SELECT MAX(TRADEDATE) FROM Futures
    """
    cursor.execute(query_max)
    max_date = cursor.fetchone()[0]
    if max_date:
        query_delete = """
            DELETE FROM Futures WHERE TRADEDATE = ?
        """
        cursor.execute(query_delete, (max_date,))
        connection.commit()
        print(f"Удалена последняя запись для даты {max_date}")
    else:
        print("Таблица Futures пуста, ничего не удалено.")

def main(
        db_path_minutes: Path = path_db_minutes,
        path_db_day: Path = path_db_day
) -> None:
    """Главная функция для конвертации минутных котировок в дневные."""
    try:
        # Подключение к базе данных с минутными котировками
        connection_minutes = sqlite3.connect(str(db_path_minutes))
        cursor_minutes = connection_minutes.cursor()

        # Получаем список уникальных дат
        dates: list = get_sorted_dates(connection_minutes, cursor_minutes, db_path_minutes)
        print(f"Найдено дат: {dates}")

        # Создаем или открываем базу данных с дневными барами
        connection_day = sqlite3.connect(str(path_db_day))
        cursor_day = connection_day.cursor()
        create_tables(connection_day)

        # Удаляем последнюю запись, если она существует
        delete_latest_record(connection_day, cursor_day)

        # Обрабатываем каждую пару дат для формирования дневных свечек
        for date_end, date_start in zip(dates, dates[1:] + ['1970-01-01']):
            start = f"{date_start} {time_start}"  # Дата и время начала дневной свечи.
            end = f"{date_end} {time_end}"  # Дата и время конца дневной свечи.

            # Получаем дневную свечку из минутных данных
            candle = get_daily_candle(cursor_minutes, start, end)
            if candle:
                save_daily_candle(connection_day, cursor_day, candle)
            else:
                print(f"Нет данных для периода {start} - {end}")

        # Закрываем соединения
        cursor_minutes.close()
        connection_minutes.close()
        cursor_day.close()
        connection_day.close()
        print("Все соединения с базами данных закрыты.")

    except sqlite3.Error as err:
        print(f"Ошибка при работе с базой данных: {err}")

if __name__ == '__main__':
    main(path_db_minutes, path_db_day)