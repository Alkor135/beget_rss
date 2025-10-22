#!/usr/bin/env python3
"""
simulate_trade.py
"""

import pickle
from pathlib import Path
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------
# Настройки (можно править)
# ----------------------
CACHE_FILE = "news_h2.pkl"
START_DATE = "2025-09-28"  # дата начала теста (включая)
TOP_N = 3  # количество ближайших похожих
PERCENTILE_THRESHOLDS = [0.75, 0.5, 0.5]  # пороги по перцентилю для топ-3
OUTPUT_FILE = "simulate_trade_results.xlsx"
EMBED_DIM = 1024  # размерность эмбеддингов bge-m3 (как было указано)
# ----------------------


def load_cache(cache_file: str):
    """Загрузить pickle с кэшем эмбеддингов."""
    if not Path(cache_file).exists():
        raise FileNotFoundError(f"Файл кэша не найден: {cache_file}")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data


def prepare_df(cache_data: list) -> pd.DataFrame:
    df = pd.DataFrame(cache_data)

    df["loaded_at"] = pd.to_datetime(df["metadata"].apply(lambda x: x["loaded_at"]))
    df["date"] = df["loaded_at"].dt.date

    df["H2"] = df["metadata"].apply(lambda x: float(x["H2"]) if x.get("H2") is not None else np.nan)
    df["Percentile"] = df["metadata"].apply(
        lambda x: float(x["Percentile"]) if x.get("Percentile") is not None else np.nan
    )

    # Преобразуем embedding в numpy-массив (если None или не список — заменим на пустой массив)
    def safe_convert(x):
        if x is None:
            return np.array([], dtype=np.float32)
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.array(x, dtype=np.float32)
        return np.array([], dtype=np.float32)

    df["embedding"] = df["embedding"].apply(safe_convert)

    # Фильтрация испорченных / пустых эмбеддингов
    before = len(df)
    df = df[df["embedding"].apply(lambda x: len(x) == EMBED_DIM)]
    removed = before - len(df)
    print(f"⚠️ Удалено {removed} записей с некорректным embedding (None, пустой или размерности != {EMBED_DIM})")

    # Фильтрация по H2 и Percentile
    before = len(df)
    df = df.dropna(subset=["H2", "Percentile"]).reset_index(drop=True)
    removed2 = before - len(df)
    print(f"⚠️ Удалено {removed2} записей с None в H2 или Percentile")

    # Нормализация L2
    def l2_normalize(vec):
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    df["embedding"] = df["embedding"].apply(l2_normalize)

    df = df.sort_values("loaded_at").reset_index(drop=True)
    return df


def simulate(df: pd.DataFrame, start_date: str):
    """
    """
    start_ts = pd.to_datetime(start_date)

    # Список уникальных дат в отсортированном порядке
    unique_dates = np.array(sorted(df["date"].unique()))

    # DF с ембеддингами для которых ищем схожие ембеддинги на истории
    df_tail = df[df["date"] >= start_ts.date()].copy()

    for idx, row in tqdm(df_tail.iterrows(), total=len(df_tail), desc="Симуляция новостей"):
        index = np.where(unique_dates == row["date"])[0][0]
        for date_df in unique_dates[:index]:
            df_date = df[df['date'] == date_df].copy()

    return df_date


def main():
    print("Загрузка кэша эмбеддингов...")
    cache_data = load_cache(CACHE_FILE)  # Загрузка кэша из pickle
    print(f"Загружено записей из кэша: {len(cache_data)}")

    df = prepare_df(cache_data)  # Подготовка DataFrame
    print(f"После фильтрации (H2/Percentile) и подготовки: {len(df)} записей")

    print("Запуск симуляции")
    results_df = simulate(df, START_DATE)
    print(results_df)
    print(results_df.columns.tolist())

    # # Преобразуем столбец loaded_at в datetime
    # results_df['loaded_at'] = pd.to_datetime(results_df['loaded_at'])
    # # Оставляем строки с временем 09:00:00 и позже
    # results_df = results_df[results_df['loaded_at'].dt.time >= pd.to_datetime('09:00:00').time()]
    # results_df['H2_cumsum'] = results_df['H2'].cumsum()
    # print(f"Сделок после фильтров: {len(results_df)}")
    #
    # # Сохранение результатов в Excel (если есть)
    # if not results_df.empty:
    #     results_df.to_excel(OUTPUT_FILE, index=False)
    #     print(f"✅ Результаты сохранены в {OUTPUT_FILE}")
    # else:
    #     print("❌ Результаты пусты — файл не создан.")


if __name__ == "__main__":
    main()
