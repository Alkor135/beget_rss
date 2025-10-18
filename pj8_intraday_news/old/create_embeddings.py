"""
Создание эмбеддингов для новостей.
Следующий бар по времени выбирается только если он не позже MAX_LAG_MINUTES.
Если нет подходящего бара → date_bar, H2, Percentile = None.
При повторном запуске обновляются только записи с None, эмбеддинг не пересоздаётся.
Эмбеддинги в формате np.float32, экономия места и скорость работы с массивами.
Асинхронные запросы к Ollama ограничены MAX_CONCURRENT_REQUESTS.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import pickle
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
import os
from datetime import timedelta

# ======================
# Настройки
# ======================
NEWS_DB_FOLDER = Path(r"C:\Users\Alkor\gd\db_rss")
FUTURES_DB_FILE = r"C:\Users\Alkor\PycharmProjects\beget_rss\pj8_intraday_news\minutes_RTS_processed_p10.sqlite"
CACHE_FILE = "news_h2.pkl"

URL_AI = 'http://localhost:11434/api/embeddings'
MODEL_NAME = 'bge-m3'

MAX_CONCURRENT_REQUESTS = 10  # ограничение одновременных запросов
UPDATE_EXISTING_METADATA = True  # обновлять metadata для старых новостей, если появились новые бары
MAX_LAG_MINUTES = 5  # максимальная разница между news_time и TRADEDATE в минутах

# ======================
# Функции
# ======================

def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

async def get_embedding(session: aiohttp.ClientSession, text: str) -> np.ndarray:
    try:
        async with session.post(URL_AI, json={'model': MODEL_NAME, 'text': text}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return np.array(data['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Ошибка при получении эмбеддинга: {e}")
        return np.array([], dtype=np.float32)

def load_news() -> pd.DataFrame:
    news_files = sorted(NEWS_DB_FOLDER.glob("rss_news_*.db"))
    df_list = []

    for db_file in news_files:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql("SELECT loaded_at, date, title, provider FROM news", conn)
        conn.close()
        df_list.append(df)

    news_df = pd.concat(df_list, ignore_index=True)
    news_df['loaded_at'] = pd.to_datetime(news_df['loaded_at'])
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df['lag'] = (news_df['loaded_at'] - news_df['date']).dt.total_seconds() / 60
    return news_df

def load_bars() -> pd.DataFrame:
    conn = sqlite3.connect(FUTURES_DB_FILE)
    bars_df = pd.read_sql("SELECT TRADEDATE, SECID, CLOSE, H2, Percentile FROM FuturesProcessed", conn)
    conn.close()
    bars_df['TRADEDATE'] = pd.to_datetime(bars_df['TRADEDATE'])
    bars_df = bars_df.sort_values('TRADEDATE').reset_index(drop=True)
    return bars_df

def find_nearest_bar(news_time, bars_df, max_lag_minutes=MAX_LAG_MINUTES):
    """
    Возвращает бар с TRADEDATE >= news_time, но не позднее max_lag_minutes.
    Если такого бара нет, возвращает None.
    """
    if len(bars_df) == 0:
        return None

    # Находим все бары >= news_time
    future_bars = bars_df[bars_df['TRADEDATE'] >= news_time]
    if future_bars.empty:
        return None

    nearest_bar = future_bars.iloc[0]

    # Проверяем, что разница <= max_lag_minutes
    delta_minutes = (nearest_bar['TRADEDATE'] - news_time).total_seconds() / 60
    if delta_minutes > max_lag_minutes:
        return None

    return nearest_bar

def load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return {entry['id']: entry for entry in data}
    return {}

def save_cache(cache_file: str, cache_dict: dict):
    with open(cache_file, 'wb') as f:
        pickle.dump(list(cache_dict.values()), f)

# ======================
# Асинхронная обработка
# ======================

async def process_news(news_subset, bars_df, cache):
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_single(row):
        news_id = md5_hash(row['title'])
        nearest_bar = find_nearest_bar(row['date'], bars_df)

        # Если новость уже есть в кэше
        if news_id in cache:
            if UPDATE_EXISTING_METADATA:
                # обновляем metadata только если ранее было None и теперь есть бар
                if cache[news_id]['metadata'].get('date_bar') is None and nearest_bar is not None:
                    cache[news_id]['metadata'].update({
                        'date_bar': nearest_bar['TRADEDATE'].strftime('%Y-%m-%d %H:%M:%S'),
                        'H2': nearest_bar['H2'],
                        'Percentile': nearest_bar['Percentile']
                    })
            return  # эмбеддинг не пересоздаём

        # создаём новый эмбеддинг
        async with sem:
            embedding = await get_embedding(session, row['title'])

        cache[news_id] = {
            'id': news_id,
            'embedding': embedding,  # numpy float32
            'metadata': {
                'loaded_at': row['loaded_at'].strftime('%Y-%m-%d %H:%M:%S'),
                'lag': row['lag'],
                'date_bar': nearest_bar['TRADEDATE'].strftime('%Y-%m-%d %H:%M:%S') if nearest_bar is not None else None,
                'H2': nearest_bar['H2'] if nearest_bar is not None else None,
                'Percentile': nearest_bar['Percentile'] if nearest_bar is not None else None
            }
        }

    async with aiohttp.ClientSession() as session:
        tasks = [process_single(row) for _, row in news_subset.iterrows()]
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            await f

# ======================
# Основной скрипт
# ======================

async def main():
    print("[INFO] Загружаем новости...")
    news_df = load_news()
    print(f"[INFO] Загружено {len(news_df)} новостей.")

    print("[INFO] Загружаем минутные бары...")
    bars_df = load_bars()
    print(f"[INFO] Загружено {len(bars_df)} баров.")

    print(f"[INFO] Загружаем существующий кэш из {CACHE_FILE}...")
    cache = load_cache(CACHE_FILE)
    print(f"[INFO] Найдено {len(cache)} существующих эмбеддингов.")

    # Фильтруем только новые новости
    news_df['news_id'] = news_df['title'].apply(md5_hash)
    news_to_process = news_df[~news_df['news_id'].isin(cache.keys())].drop(columns='news_id')
    print(f"[INFO] Новых новостей для обработки: {len(news_to_process)}")

    if len(news_to_process) > 0 or UPDATE_EXISTING_METADATA:
        # обрабатываем весь набор новостей, чтобы обновить None метаданные
        await process_news(news_df, bars_df, cache)
        print(f"[INFO] Сохраняем кэш в {CACHE_FILE}...")
        save_cache(CACHE_FILE, cache)
        print("[INFO] Готово!")
    else:
        print("[INFO] Новых новостей нет. Кэш актуален.")

if __name__ == "__main__":
    asyncio.run(main())
