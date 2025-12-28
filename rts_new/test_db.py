"""
Тестирует БД SQLite, проверяя существование файла и выводя список таблиц.
"""

import os
import sqlite3

# Путь к базе данных
db_path = r'C:/Users/Alkor/gd/data_quote_db/RTS_futures_day_2025_21-00.db'

# Проверяем существование файла
if not os.path.exists(db_path):
    print("Файл базы данных не найден:", db_path)
else:
    print("Файл найден. Подключение к базе данных...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Запрос на получение всех таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if tables:
            print("Найдены таблицы:")
            for table in tables:
                print(f"  - {table[0]}")
        else:
            print("Таблиц в базе данных не обнаружено.")

        conn.close()
    except sqlite3.Error as e:
        print("Ошибка при работе с SQLite:", e)

