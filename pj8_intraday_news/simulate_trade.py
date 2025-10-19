#!/usr/bin/env python3
"""
simulate_trade.py

Симуляция торговли на основе эмбеддингов новостей по новым правилам.

Requirements:
    pip install pandas numpy openpyxl scipy tqdm
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine
from tqdm import tqdm

# ----------------------
# Настройки
# ----------------------
CACHE_FILE = "news_h2.pkl"
START_DATE = "2025-10-01"  # дата начала теста
TOP_N = 3  # количество ближайших похожих
PERCENTILE_THRESHOLDS = [0.95, 0.95, 0.95]  # для ближайших эмбеддингов
OUTPUT_FILE = "simulate_trade_results.xlsx"


# ----------------------
# Загрузка кэша
# ----------------------
def load_cache(cache_file: str):
    if not Path(cache_file).exists():
        raise FileNotFoundError(f"Файл кэша не найден: {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    return data


# ----------------------
# Подготовка DataFrame
# ----------------------
def prepare_df(cache_data):
    df = pd.DataFrame(cache_data)
    df['loaded_at'] = pd.to_datetime(df['metadata'].apply(lambda x: x['loaded_at']))

    # Приводим H2 и Percentile к float
    df['H2'] = df['metadata'].apply(lambda x: float(x['H2']) if x['H2'] is not None else np.nan)
    df['Percentile'] = df['metadata'].apply(
        lambda x: float(x['Percentile']) if x['Percentile'] is not None else np.nan)

    df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))

    # Исключаем записи с None в H2 или Percentile
    df = df.dropna(subset=['H2', 'Percentile'])

    # Сортировка по дате
    df = df.sort_values('loaded_at').reset_index(drop=True)
    return df


# ----------------------
# Косинусная схожесть
# ----------------------
def cosine_similarity(a, b):
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return 1 - cosine(a, b)


# ----------------------
# Симуляция торговли
# ----------------------
def simulate(df, start_date: str):
    start_ts = pd.to_datetime(start_date)
    results = []

    past_embeddings = []
    past_H2 = []
    past_percentiles = []
    past_loaded_at = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Симуляция новостей"):
        if row['loaded_at'] < start_ts:
            past_embeddings.append(row['embedding'])
            past_H2.append(row['H2'])
            past_percentiles.append(row['Percentile'])
            past_loaded_at.append(row['loaded_at'])
            continue

        if len(past_embeddings) < TOP_N:
            # Сохраняем эмбеддинг, но не обрабатываем, т.к. недостаточно прошлых
            past_embeddings.append(row['embedding'])
            past_H2.append(row['H2'])
            past_percentiles.append(row['Percentile'])
            past_loaded_at.append(row['loaded_at'])
            continue

        # Косинусное сходство с предыдущими
        sims = [cosine_similarity(row['embedding'], emb) for emb in past_embeddings]
        sims_sorted_idx = np.argsort(sims)[::-1]  # от наибольшей похожести
        top_idx = [i for i in sims_sorted_idx if past_H2[i] is not None][:TOP_N]

        if len(top_idx) < TOP_N:
            past_embeddings.append(row['embedding'])
            past_H2.append(row['H2'])
            past_percentiles.append(row['Percentile'])
            past_loaded_at.append(row['loaded_at'])
            continue

        # Берем top эмбеддинги
        top_H2 = [past_H2[i] for i in top_idx]
        top_percentile = [past_percentiles[i] for i in top_idx]

        # Проверка порогов Percentile
        if top_percentile[0] > PERCENTILE_THRESHOLDS[0] and \
                top_percentile[1] > PERCENTILE_THRESHOLDS[1] and \
                top_percentile[2] > PERCENTILE_THRESHOLDS[2]:

            # Все три H2 положительные
            if all(h > 0 for h in top_H2):
                if row['H2'] > 0:
                    results.append({'loaded_at': row['loaded_at'], 'H2': abs(row['H2']), 'dir': 'buy'})  #
                elif row['H2'] < 0:
                    results.append({'loaded_at': row['loaded_at'], 'H2': -abs(row['H2']), 'dir': 'buy'})

            # Все три H2 отрицательные
            elif all(h < 0 for h in top_H2):
                if row['H2'] < 0:
                    results.append({'loaded_at': row['loaded_at'], 'H2': abs(row['H2']), 'dir': 'sell'})  #
                elif row['H2'] > 0:
                    results.append({'loaded_at': row['loaded_at'], 'H2': -abs(row['H2']), 'dir': 'sell'})

        # Сохраняем текущий эмбеддинг для будущих сравнений
        past_embeddings.append(row['embedding'])
        past_H2.append(row['H2'])
        past_percentiles.append(row['Percentile'])
        past_loaded_at.append(row['loaded_at'])

    # Превращаем в DataFrame и убираем дубликаты
    # result_df = pd.DataFrame(results).drop_duplicates(subset=['loaded_at'])
    result_df = pd.DataFrame(results)
    return result_df


# ----------------------
# Main
# ----------------------
def main():
    print("Загрузка кэша эмбеддингов...")
    cache_data = load_cache(CACHE_FILE)
    print(f"Загружено {len(cache_data)} записей")

    df = prepare_df(cache_data)
    print(f"После фильтрации и сортировки: {len(df)} записей")

    print("Запуск симуляции торговли...")
    results_df = simulate(df, START_DATE)
    print(f"Количество сделок после фильтров: {len(results_df)}")

    #  === Очистка результатов 2 часа ===
    results_df['loaded_at'] = pd.to_datetime(results_df['loaded_at'])
    # Сортировка по возрастанию (от ранних к поздним)
    results_df = results_df.sort_values(by='loaded_at', ascending=True)
    # Расчёт разницы между соседними строками
    results_df['time_diff'] = results_df['loaded_at'].diff()
    # Фильтрация строк: оставляем первую и те, где разница ≥ 2 часа
    results_df = results_df[
        (results_df['time_diff'].isna()) | (results_df['time_diff'] >= pd.Timedelta(hours=2))]
    # Удаляем временный столбец, если больше не нужен
    results_df = results_df.drop(columns=['time_diff'])

    results_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Результаты сохранены в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
