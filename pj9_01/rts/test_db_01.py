import sqlite3
import pandas as pd
from pathlib import Path

# Настройки для отображения широкого df pandas
pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

db_paths = [
    r"C:/Users/Alkor/gd/data_quote_db/RTS_futures_minute_2025.db",
    r"C:/Users/Alkor/gd/data_quote_db/minutes_RTS_2025.sqlite"
]

def show_head_tail_sorted(db_path):
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name",
        conn
    )['name']
    for table in tables:
        print(f"\n=== {db_path} :: {table} ===")
        # Забираем всю таблицу или достаточно строк
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        if 'TRADEDATE' in df.columns:
            # Преобразуем к datetime и сортируем
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], format="%Y-%m-%d %H:%M:%S")
            df = df.sort_values('TRADEDATE')
            print("First 5 rows:")
            print(df.head(5))
            print("Last 5 rows:")
            print(df.tail(5))
        else:
            print("No TRADEDATE column found in this table, skipping sorting.")
    conn.close()

for path in db_paths:
    show_head_tail_sorted(path)

df = pd.read_pickle(Path('minutes_RTS_processed_p10.pkl'))
print(df)

na_count = df["Percentile"].isna().sum()
total = len(df)
na_share = na_count / total if total else 0
print(f'Для колонки Percentile: {na_count=}, {total=}, {na_share=}')

print(f'\nДля колонки H2_abs доля: {df["H2_abs"].isna().mean()}')

print("\nДоля NA в H2_abs:", df["H2_abs"].isna().mean())
print("Доля NA в Percentile:", df["Percentile"].isna().mean())

df["TRADEDATE_DATE"] = df["TRADEDATE"].dt.date
print("\nУникальных дат:", df["TRADEDATE_DATE"].nunique())
print("Минимум/максимум дат:", df["TRADEDATE_DATE"].min(), df["TRADEDATE_DATE"].max())

