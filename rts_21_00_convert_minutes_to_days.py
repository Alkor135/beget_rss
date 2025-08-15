"""
Конвертирует минутные котировки в дневные с заданным временем начала и конца дневной свечи.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

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
                    CLOSE REAL NOT NULL
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

    Args:
        cursor: Курсор SQLite для выполнения запросов.
        start: Начало периода в формате 'YYYY-MM-DD HH:MM:SS'.
        end: Конец периода в формате 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        tuple: (TRADEDATE, OPEN, LOW, HIGH, CLOSE) или None, если данных нет.
    """
    # Извлекаем дату из end для TRADEDATE
    trade_date = datetime.strptime(end, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

    # Запрос для получения OPEN, LOW, HIGH, CLOSE
    query = """
        SELECT 
            (SELECT OPEN FROM Futures WHERE TRADEDATE = (
                SELECT MIN(TRADEDATE) FROM Futures WHERE TRADEDATE BETWEEN ? AND ?
            ) LIMIT 1) AS OPEN,
            MIN(LOW) AS LOW,
            MAX(HIGH) AS HIGH,
            (SELECT CLOSE FROM Futures WHERE TRADEDATE = (
                SELECT MAX(TRADEDATE) FROM Futures WHERE TRADEDATE BETWEEN ? AND ?
            ) LIMIT 1) AS CLOSE
        FROM Futures
        WHERE TRADEDATE BETWEEN ? AND ?
    """
    cursor.execute(query, (start, end, start, end, start, end))
    result = cursor.fetchone()

    if result and result[0] is not None and result[1] is not None and result[2] is not None and result[3] is not None:
        return (trade_date, result[0], result[1], result[2], result[3])
    return None

def save_daily_candle(connection: sqlite3.Connection, cursor, candle: tuple) -> None:
    """
    Сохраняет дневную свечку в таблицу Futures.

    Args:
        connection: Соединение с базой данных.
        cursor: Курсор SQLite для выполнения запросов.
        candle: Кортеж (TRADEDATE, OPEN, LOW, HIGH, CLOSE).
    """
    if candle:
        query = """
            INSERT INTO Futures (TRADEDATE, OPEN, LOW, HIGH, CLOSE)
            VALUES (?, ?, ?, ?, ?)
        """
        with connection:
            try:
                cursor.execute(query, candle)
                connection.commit()
                print(f"Сохранена дневная свечка для {candle[0]}")
            except sqlite3.Error as e:
                print(f"Ошибка при сохранении дневной свечки: {e}")

def main(db_path_minutes: Path, path_db_day: Path) -> None:
    """Главная функция для конвертации минутных котировок в дневные."""
    try:
        # Подключение к базе данных с минутными котировками
        connection_minutes = sqlite3.connect(str(db_path_minutes))
        cursor_minutes = connection_minutes.cursor()

        # Получаем список уникальных дат
        dates: list = get_sorted_dates(connection_minutes, cursor_minutes, db_path_minutes)
        print(f"Найдено дат: {dates}")

        # Удаляем старую базу данных с дневными барами с 21:00 предыдущей сессии (если она существует)
        if path_db_day.exists():
            try:
                path_db_day.unlink()
                print(f"Старая база данных {path_db_day} удалена.")
            except OSError as e:
                print(f"Ошибка при удалении базы данных {path_db_day}: {e}")

        # Создаем новую базу данных и таблицу Futures
        connection_day = sqlite3.connect(str(path_db_day))
        cursor_day = connection_day.cursor()
        create_tables(connection_day)

        # Обрабатываем каждую пару дат для формирования дневных свечек
        for e, s in zip(dates, dates[1:] + ['1970-01-01']):
            start = f"{s} 21:00:00"
            end = f"{e} 20:59:00"

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

    except sqlite3.Error as e:
        print(f"Ошибка при работе с базой данных: {e}")

if __name__ == '__main__':
    ticker = 'RTS'
    # Путь к файлу БД c минутными котировками скаченных с MOEX ISS
    path_db_minutes: Path = Path(
        rf'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_minute_2025.db'
    )
    # Путь к файлу БД с дневными котировками (с 21:00 предыдущей сессии)
    path_db_day: Path = Path(
        rf'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db'
    )

    main(path_db_minutes, path_db_day)
