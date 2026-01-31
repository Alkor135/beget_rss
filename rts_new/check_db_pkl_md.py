import sqlite3
import pandas as pd

# === ПРОВЕРКА БД С ДНЕВНЫМИ КОТИРОВКАМИ С 21:00 ===
# Путь к базе данных
db_path = 'C:/Users/Alkor/gd/data_quote_db/rts_futures_day_2025_21-00.db'

# Подключение к БД и чтение данных в DataFrame
conn = sqlite3.connect(db_path)
query = "SELECT * FROM Futures ORDER BY TRADEDATE ASC"
df = pd.read_sql_query(query, conn)

# Закрытие соединения
conn.close()

# Вывод DataFrame в консоль
print("Датафрейм с дневными котировками:")
print(df)

# === ПРОВЕРКА PKL ФАЙЛА С ЕМБЕДДИНГАМИ ===
# Путь к pkl файлу
pkl_path = 'embeddings_ollama.pkl'

# Загрузка DataFrame из pkl файла
df_embeddings = pd.read_pickle(pkl_path)

# Вывод DataFrame в консоль
with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 100
):
    print("Датафрейм с эмбеддингами:")
    print(df_embeddings)
