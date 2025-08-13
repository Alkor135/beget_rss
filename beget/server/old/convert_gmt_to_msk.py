import sqlite3
import os
from datetime import datetime, timedelta

# Папки
source_dir = "/home/user/rss_scraper/db_data"
target_dir = "/home/user/rss_scraper/db_data_investing"

# Создаем целевую папку, если она не существует
os.makedirs(target_dir, exist_ok=True)

# Список файлов баз данных
db_files = [
    "rss_news_investing_2025_06.db",
    "rss_news_investing_2025_07.db",
    "rss_news_investing_2025_08.db"
]

# Функция для преобразования времени GMT в MSK
def gmt_to_msk(date_str):
    try:
        # Парсим дату из строки
        gmt_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        # Добавляем 3 часа для MSK
        msk_time = gmt_time + timedelta(hours=3)
        # Форматируем обратно в строку
        return msk_time.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        print(f"Ошибка формата даты: {date_str}, {e}")
        return date_str  # Возвращаем исходную строку, если ошибка

# Обрабатываем каждый файл
for db_file in db_files:
    source_path = os.path.join(source_dir, db_file)
    target_path = os.path.join(target_dir, db_file)

    # Подключаемся к исходной базе данных
    conn_source = sqlite3.connect(source_path)
    cursor_source = conn_source.cursor()

    # Предполагаем, что таблица называется 'news' и имеет поле 'date'
    # Получаем структуру таблицы
    cursor_source.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_name = cursor_source.fetchone()[0]  # Берем первую таблицу
    print(f"Обрабатываем таблицу {table_name} в {db_file}")

    # Получаем все записи
    cursor_source.execute(f"SELECT * FROM {table_name}")
    rows = cursor_source.fetchall()

    # Получаем названия столбцов
    cursor_source.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor_source.fetchall()]
    date_index = columns.index('date')  # Индекс столбца 'date'

    # Создаем новую базу данных
    conn_target = sqlite3.connect(target_path)
    cursor_target = conn_target.cursor()

    # Создаем таблицу с той же структурой
    cursor_source.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    create_table_sql = cursor_source.fetchone()[0]
    cursor_target.execute(create_table_sql)

    # Подготовка для вставки данных
    placeholders = ','.join(['?' for _ in columns])
    insert_sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"

    # Обрабатываем и записываем данные
    for row in rows:
        row_list = list(row)
        # Преобразуем поле date
        row_list[date_index] = gmt_to_msk(row_list[date_index])
        cursor_target.execute(insert_sql, row_list)

    # Сохраняем изменения и закрываем соединения
    conn_target.commit()
    conn_source.close()
    conn_target.close()
    print(f"Обработан файл {db_file}, сохранен в {target_path}")

print("Все файлы обработаны.")