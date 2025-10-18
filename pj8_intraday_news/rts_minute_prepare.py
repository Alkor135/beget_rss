#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import glob
import logging

# =======================
# CONFIGURATION
# =======================
SOURCE_DIR = r"C:\Users\Alkor\gd\data_quote_db"
SOURCE_MASK = "minutes_RTS_*.sqlite"
LOOKBACK_DAYS = 10  # количество предыдущих торговых дней для Percentile
TARGET_DB = f"minutes_RTS_processed_p{LOOKBACK_DAYS}.sqlite"
# LOG_FILE = Path(SOURCE_DIR) / "process_quotes.log"
LOG_FILE = "process_quotes.log"

# =======================
# LOGGING SETUP
# =======================
logger = logging.getLogger("process_quotes")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# file handler
fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)


# =======================
# FUNCTIONS
# =======================
def create_target_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS FuturesProcessed (
        TRADEDATE TEXT PRIMARY KEY,
        SECID TEXT,
        OPEN REAL,
        LOW REAL,
        HIGH REAL,
        CLOSE REAL,
        VOLUME INTEGER,
        LSTTRADE TEXT,
        H2 REAL,
        H2_abs REAL,
        Percentile REAL
    )
    """)
    conn.commit()


def process_H2(df):
    # Рассчитываем TRADEDATE + 2H
    df["TRADEDATE_2h"] = df["TRADEDATE"] + timedelta(hours=2)
    df_future = df[["TRADEDATE", "SECID", "CLOSE"]].rename(
        columns={"TRADEDATE": "TRADEDATE_future", "CLOSE": "CLOSE_future"})
    df = df.merge(df_future, left_on=["TRADEDATE_2h", "SECID"],
                  right_on=["TRADEDATE_future", "SECID"], how="left")
    df["H2"] = df["CLOSE_future"] - df["OPEN"]
    df["H2_abs"] = df["H2"].abs()
    df.drop(columns=["TRADEDATE_2h", "TRADEDATE_future", "CLOSE_future"], inplace=True)
    return df


def compute_percentile(df, lookback_days):
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    df["Percentile"] = None

    # Получаем уникальные даты с котировками
    df["TRADEDATE_DATE"] = df["TRADEDATE"].dt.date
    unique_dates = df["TRADEDATE_DATE"].drop_duplicates().tolist()
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}

    # tqdm
    pbar = tqdm(total=len(df), desc="Computing Percentile")

    # Индекс для каждой строки
    for idx, row in df.iterrows():
        current_date = row["TRADEDATE_DATE"]
        current_idx = date_to_idx[current_date]
        start_idx = max(0, current_idx - lookback_days)
        lookback_dates = unique_dates[start_idx:current_idx]
        if not lookback_dates:
            df.at[idx, "Percentile"] = None
        else:
            pool = df[df["TRADEDATE_DATE"].isin(lookback_dates)]["H2_abs"].dropna()
            if len(pool) == 0:
                df.at[idx, "Percentile"] = None
            else:
                rank = (pool <= row["H2_abs"]).sum()
                df.at[idx, "Percentile"] = rank / len(pool)
        pbar.update(1)
    pbar.close()
    df.drop(columns=["TRADEDATE_DATE"], inplace=True)
    return df


# =======================
# MAIN
# =======================
def main():
    target_path = Path(TARGET_DB)
    create_target = not target_path.exists()

    # Создаём соединение с целевой БД
    with sqlite3.connect(TARGET_DB) as conn_target:
        create_target_table(conn_target)

        # Обрабатываем каждый исходный файл
        source_files = sorted(glob.glob(str(Path(SOURCE_DIR) / SOURCE_MASK)))
        for src_file in source_files:
            logger.info(f"Processing source DB: {src_file}")
            with sqlite3.connect(src_file) as conn_src:
                df_src = pd.read_sql("SELECT * FROM Futures", conn_src,
                                     parse_dates=["TRADEDATE", "LSTTRADE"])

                # Оставляем только новые записи
                existing_dates = \
                pd.read_sql("SELECT TRADEDATE FROM FuturesProcessed", conn_target)["TRADEDATE"]
                df_src = df_src[~df_src["TRADEDATE"].astype(str).isin(existing_dates)]
                if df_src.empty:
                    logger.info(f"No new bars to process in {src_file}")
                    continue

                # Проход 1: H2 / H2_abs
                logger.info(f"Calculating H2 / H2_abs for {len(df_src)} bars")
                df_src = process_H2(df_src)

                # Сохраняем промежуточно (Percentile пока None)
                df_src.to_sql("FuturesProcessed", conn_target, if_exists="append", index=False)
                logger.info(f"H2 / H2_abs saved to {TARGET_DB}")

        # Проход 2: Percentile
        logger.info(f"Starting Percentile computation with lookback {LOOKBACK_DAYS} days")
        df_all = pd.read_sql("SELECT * FROM FuturesProcessed", conn_target,
                             parse_dates=["TRADEDATE", "LSTTRADE"])
        df_all = compute_percentile(df_all, LOOKBACK_DAYS)

        # Сохраняем Percentile
        logger.info(f"Updating Percentile in target DB")
        df_all.to_sql("FuturesProcessed", conn_target, if_exists="replace", index=False)

        # VACUUM
        logger.info(f"Performing VACUUM on {TARGET_DB}")
        conn_target.execute("VACUUM")
        logger.info(f"Processing completed successfully")


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info("===== Starting process_quotes.py =====")
    main()
    end_time = datetime.now()
    logger.info(f"===== Finished. Total time: {end_time - start_time} =====")
