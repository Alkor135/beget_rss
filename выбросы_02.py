"""
Скрипт анализирует минутные свечи фьючерсов на индекс РТС за период с 2020 по 2025 год.
Он вычисляет абсолютное изменение цены за 15 минут как разницу между закрытием 15-й свечи и открытием текущей.
Для каждой даты собираются аналогичные изменения из 10 предыдущих торговых дней.
На основе этих данных строится распределение, и с помощью IQR определяются верхние выбросы.
Фиксируются только те свечи, где 15-минутное изменение значительно превышает исторический фон.
Результат — список потенциально аномальных свечей с резкими движениями вверх.
Итог сохраняется в CSV для дальнейшего анализа.
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import timedelta
from tqdm import tqdm

# Шаг 1: Загрузка данных из БД
db_path = r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_minute_2025.db'
conn = sqlite3.connect(db_path)
query = "SELECT TRADEDATE, OPEN, HIGH, LOW, CLOSE FROM Futures ORDER BY TRADEDATE"
df = pd.read_sql_query(query, conn, parse_dates=['TRADEDATE'])
conn.close()

# Проверяем, что TRADEDATE — datetime
if not pd.api.types.is_datetime64_any_dtype(df['TRADEDATE']):
    raise ValueError("Колонка TRADEDATE не распознана как datetime. Проверьте формат.")

# Добавляем колонку 'date' для группировки по дням
df['date'] = df['TRADEDATE'].dt.date

# Шаг 2: Векторизованный расчёт 15-минутных разниц
df['15min_change'] = abs(df['CLOSE'].shift(-14) - df['OPEN'])  # |close(t+14) - open(t)|

# Шаг 3: Функция для сбора исторических разниц за 10 предыдущих дней
def get_historical_changes(df, current_date, date_list):
    current_idx = date_list.index(current_date)
    prev_dates = date_list[max(0, current_idx - 10):current_idx]  # Последние 10 дней
    if len(prev_dates) < 10:
        return None
    hist_df = df[df['date'].isin(prev_dates)]
    hist_changes = abs(hist_df['CLOSE'].shift(-14) - hist_df['OPEN']).dropna().values
    return hist_changes if len(hist_changes) >= 10 else None

# Шаг 4: Проверка на выброс (только большие значения)
def is_outlier(value, hist_changes):
    if hist_changes is None or len(hist_changes) < 10:
        return False
    q1 = np.percentile(hist_changes, 25)
    q3 = np.percentile(hist_changes, 75)
    iqr = q3 - q1
    return value > q3 + 1.5 * iqr  # Только верхняя граница

# Шаг 5: Поиск свечей с выбросами
unique_dates = sorted(df['date'].unique())  # Все уникальные даты
outlier_candles = []

# Прогресс-бар для основного цикла
for idx in tqdm(range(len(df) - 14), desc="Обработка свечей"):
    if np.isnan(df.loc[idx, '15min_change']):
        continue
    current_date = df.loc[idx, 'date']
    change = df.loc[idx, '15min_change']
    hist_changes = get_historical_changes(df, current_date, unique_dates)
    if is_outlier(change, hist_changes):
        outlier_candles.append({
            'TRADEDATE': df.loc[idx, 'TRADEDATE'],
            'OPEN': df.loc[idx, 'OPEN'],
            'CLOSE': df.loc[idx, 'CLOSE'],
            '15min_change': change
        })

# Результат: DataFrame с такими свечами
result_df = pd.DataFrame(outlier_candles)
print(result_df.head(10))
print(len(result_df))

# Сохраняем результат в CSV
# result_df.to_csv('outlier_candles_large_changes.csv', index=False)
# print("Результат сохранён в 'outlier_candles_large_changes.csv'")