#!/usr/bin/env python3
"""
create_embeddings.py

Асинхронный сбор эмбеддингов заголовков новостей (Ollama) с дедупликацией по (title + loaded_at.date),
привязкой к минутным барам (TRADEDATE >= loaded_at, не позже MAX_LAG_MINUTES), сохранением в news_h2.pkl.

Консоль: print + tqdm прогресс
Лог-файл: cache_builder.log (только логирование, разделение по уровням)

Requirements:
    pip install aiohttp tqdm pandas numpy
"""

import os
import sqlite3
import hashlib
import pickle
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import aiohttp
from tqdm.asyncio import tqdm_asyncio
import logging

# ----------------------
# Настройки
# ----------------------
NEWS_DB_FOLDER = Path(r"C:\Users\Alkor\gd\db_rss")
FUTURES_DB_FILE = r"C:\Users\Alkor\PycharmProjects\beget_rss\pj8_intraday_news\minutes_RTS_processed_p10.sqlite"
CACHE_FILE = "news_h2.pkl"

URL_AI = 'http://localhost:11434/api/embeddings'
MODEL_NAME = 'bge-m3'

MAX_CONCURRENT_REQUESTS = 10
MAX_LAG_MINUTES = 5
UPDATE_EXISTING_METADATA = True
SAVE_EVERY = 200

LOG_FILE = "cache_builder.log"

# ----------------------
# Логирование в файл
# ----------------------
logger = logging.getLogger("cache_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ----------------------
# Утилиты
# ----------------------
def md5_hash_text(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def news_id_from_title_and_loaded_at(title: str, loaded_at_ts: pd.Timestamp) -> str:
    date_only = loaded_at_ts.strftime("%Y-%m-%d")
    return md5_hash_text(f"{title}_{date_only}")

# ----------------------
# Загрузка данных
# ----------------------
def load_news() -> pd.DataFrame:
    news_files = sorted(NEWS_DB_FOLDER.glob("rss_news_*.db"))
    df_list = []
    for db_file in news_files:
        try:
            conn = sqlite3.connect(db_file)
            df = pd.read_sql("SELECT loaded_at, date, title, provider FROM news", conn)
            conn.close()
            df_list.append(df)
        except Exception as e:
            logger.error(f"Ошибка чтения {db_file}: {e}")
    if not df_list:
        return pd.DataFrame(columns=['loaded_at','date','title','provider'])
    news_df = pd.concat(df_list, ignore_index=True)
    news_df['loaded_at'] = pd.to_datetime(news_df['loaded_at'])
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df['lag'] = (news_df['loaded_at'] - news_df['date']).dt.total_seconds() / 60.0
    return news_df

def load_bars() -> pd.DataFrame:
    if not os.path.exists(FUTURES_DB_FILE):
        logger.warning(f"Файл баров не найден: {FUTURES_DB_FILE}")
        return pd.DataFrame(columns=['TRADEDATE','OPEN','H2','Percentile'])
    try:
        conn = sqlite3.connect(FUTURES_DB_FILE)
        bars_df = pd.read_sql("SELECT TRADEDATE, OPEN, H2, Percentile FROM FuturesProcessed", conn)
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка чтения bars DB: {e}")
        return pd.DataFrame(columns=['TRADEDATE','OPEN','H2','Percentile'])
    bars_df['TRADEDATE'] = pd.to_datetime(bars_df['TRADEDATE'])
    bars_df = bars_df.sort_values('TRADEDATE').reset_index(drop=True)
    return bars_df

# ----------------------
# Поиск ближайшего бара
# ----------------------
def find_nearest_bar(loaded_at: pd.Timestamp, bars_df: pd.DataFrame, max_lag_minutes: int = MAX_LAG_MINUTES) -> Optional[pd.Series]:
    if bars_df is None or bars_df.empty:
        return None
    future_bars = bars_df[bars_df['TRADEDATE'] >= loaded_at]
    if future_bars.empty:
        return None
    nearest_bar = future_bars.iloc[0]
    delta_min = (nearest_bar['TRADEDATE'] - loaded_at).total_seconds() / 60.0
    if delta_min > max_lag_minutes:
        return None
    return nearest_bar

# ----------------------
# Кэш
# ----------------------
def load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return {entry['id']: entry for entry in data}
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша {cache_file}: {e}")
            return {}
    return {}

def save_cache(cache_file: str, cache_dict: dict):
    tmp = cache_file + ".tmp"
    try:
        with open(tmp, 'wb') as f:
            pickle.dump(list(cache_dict.values()), f)
        os.replace(tmp, cache_file)
        logger.info(f"Кэш сохранён: {cache_file} (записей: {len(cache_dict)})")
    except Exception as e:
        logger.error(f"Ошибка сохранения кэша: {e}")

# ----------------------
# Асинхронный запрос эмбеддинга
# ----------------------
async def fetch_embedding(session: aiohttp.ClientSession, text: str) -> np.ndarray:
    try:
        payload = {'model': MODEL_NAME, 'text': text}
        async with session.post(URL_AI, json=payload, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()
            emb = np.array(data.get('embedding', []), dtype=np.float32)
            return emb
    except Exception as e:
        logger.error(f"Ошибка получения эмбеддинга для текста: {text[:100]}... {e}")
        return np.array([], dtype=np.float32)

# ----------------------
# Асинхронная обработка новостей
# ----------------------
async def process_news_async(news_df: pd.DataFrame, bars_df: pd.DataFrame, cache: dict):
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    created = 0
    updated = 0
    processed_since_save = 0

    async def process_row(session: aiohttp.ClientSession, row: pd.Series):
        nonlocal created, updated, processed_since_save
        title = row['title']
        loaded_at = row['loaded_at']
        lag = row.get('lag', None)
        nid = news_id_from_title_and_loaded_at(title, loaded_at)
        nearest_bar = find_nearest_bar(loaded_at, bars_df, MAX_LAG_MINUTES)

        # обновление существующей записи
        if nid in cache:
            if UPDATE_EXISTING_METADATA:
                meta = cache[nid].get('metadata', {})
                if (meta.get('date_bar') is None) and (nearest_bar is not None):
                    meta.update({
                        'date_bar': nearest_bar['TRADEDATE'].strftime("%Y-%m-%d %H:%M:%S"),
                        'H2': nearest_bar['H2'],
                        'Percentile': nearest_bar['Percentile'],
                        'OPEN': nearest_bar['OPEN']
                    })
                    cache[nid]['metadata'] = meta
                    print(f"Обновлены метаданные для существующего id {nid} ({loaded_at.date()})")
                    updated += 1
                    processed_since_save += 1
            return

        # создание нового эмбеддинга
        async with sem:
            emb = await fetch_embedding(session, title)
        meta = {
            'loaded_at': loaded_at.strftime("%Y-%m-%d %H:%M:%S"),
            'lag': lag,
            'date_bar': nearest_bar['TRADEDATE'].strftime("%Y-%m-%d %H:%M:%S") if nearest_bar is not None else None,
            'H2': nearest_bar['H2'] if nearest_bar is not None else None,
            'Percentile': nearest_bar['Percentile'] if nearest_bar is not None else None,
            'OPEN': nearest_bar['OPEN'] if nearest_bar is not None else None
        }
        cache[nid] = {'id': nid, 'embedding': emb, 'metadata': meta}
        created += 1
        processed_since_save += 1
        # print(f"Создан эмбеддинг для id {nid} ({loaded_at.date()})")

    async with aiohttp.ClientSession() as session:
        tasks = [process_row(session, row) for _, row in news_df.iterrows()]
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            try:
                await fut
            except Exception:
                logger.exception("Ошибка при обработке одной задачи")
            if processed_since_save >= SAVE_EVERY:
                save_cache(CACHE_FILE, cache)
                processed_since_save = 0

    save_cache(CACHE_FILE, cache)
    print(f"Обработка завершена. Создано: {created}, Обновлено: {updated}")
    return created, updated

# ----------------------
# Дедупликация
# ----------------------
def deduplicate_news(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return news_df
    df = news_df.copy()
    df['loaded_date'] = df['loaded_at'].dt.strftime("%Y-%m-%d")
    df = df.sort_values('loaded_at')
    dedup = df.drop_duplicates(subset=['title', 'loaded_date'], keep='first')
    dedup = dedup.drop(columns=['loaded_date']).reset_index(drop=True)
    return dedup

# ----------------------
# Main
# ----------------------
async def main():
    print("Запуск скрипта create_embeddings")
    news_df = load_news()
    print(f"Загружено новостей: {len(news_df)}")
    if news_df.empty:
        print("Нет новостей для обработки. Выход.")
        return

    bars_df = load_bars()
    print(f"Загружено баров: {len(bars_df)}")

    dedup_df = deduplicate_news(news_df)
    print(f"После дедупликации: {len(dedup_df)} строк (ранние новости по title + дате)")

    cache = load_cache(CACHE_FILE)
    print(f"Загружено записей кэша: {len(cache)}")

    dedup_df['news_id'] = dedup_df.apply(lambda r: news_id_from_title_and_loaded_at(r['title'], r['loaded_at']), axis=1)
    to_create_df = dedup_df[~dedup_df['news_id'].isin(cache.keys())].drop(columns=['news_id'])
    print(f"Новых новостей для создания эмбеддингов: {len(to_create_df)}")

    await process_news_async(dedup_df, bars_df, cache)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Прервано пользователем")
    except Exception as e:
        logger.exception(f"Фатальная ошибка: {e}")
