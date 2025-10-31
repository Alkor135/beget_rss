#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# minute_prepare_pkl.py
"""
Объединяет минутные данные по фьючерсу RTS из нескольких SQLite-баз и сохраняет
результат в PKL. Поле SECID полностью исключено. Уникальность строк обеспечивается
ключом TRADEDATE (в исходных БД поле уникально).

Рассчитывается:
- H2 — изменение цены через 2 часа после открытия;
- H2_abs — абсолютное значение H2;
- Percentile — процентиль H2_abs относительно предыдущих LOOKBACK_DAYS торговых дней.

Особенности:
- Инкрементальная обработка: читается существующий PKL, добавляются только новые TRADEDATE;
- Percentile пересчитывается только для записей без значения, с узким контекстом.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import glob
import logging
import yaml

# =======================
# НАСТРОЙКИ И ЛОГИ
# =======================

SETTINGS_FILE = Path(__file__).parent / "settings_rts.yaml"
with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)

TICKER = settings.get("ticker", "RTS")
SOURCE_DIR = Path(settings.get("path_db_dir", ""))              # Папка с исходными SQLite
SOURCE_MASK = settings["path_db_min_file"].replace("{ticker}", TICKER)  # Маска файлов
LOOKBACK_DAYS = int(settings.get("lookback_days", 10))

TARGET_PKL = f"minutes_{TICKER}_processed_p{LOOKBACK_DAYS}.pkl"
LOG_FILE = "minute_prepare.log"

logger = logging.getLogger("minute_prepare_pkl")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# =======================
# ВСПОМОГАТЕЛЬНЫЕ
# =======================

def ensure_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def process_H2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает H2 и H2_abs по минутным данным.
    Требуемые столбцы: TRADEDATE, OPEN, CLOSE.
    """
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    df["TRADEDATE_2h"] = df["TRADEDATE"] + timedelta(hours=2)

    # Будущие CLOSE через 2 часа по совпадению времени
    future = df[["TRADEDATE", "CLOSE"]].rename(
        columns={"TRADEDATE": "TRADEDATE_future", "CLOSE": "CLOSE_future"}
    )
    df = df.merge(
        future,
        left_on="TRADEDATE_2h",
        right_on="TRADEDATE_future",
        how="left",
    )

    df["H2"] = df["CLOSE_future"] - df["OPEN"]
    df["H2_abs"] = df["H2"].abs()
    df.drop(columns=["TRADEDATE_2h", "TRADEDATE_future", "CLOSE_future"], inplace=True)
    return df

def compute_percentile(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """
    Вычисляет Percentile для H2_abs относительно предыдущих lookback_days торговых дней.
    Возвращает df с заполненным столбцом Percentile (в диапазоне [0,1]).
    """
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    if "Percentile" not in df.columns:
        df["Percentile"] = pd.NA

    df["__DATE__"] = df["TRADEDATE"].dt.date
    unique_dates = df["__DATE__"].drop_duplicates().tolist()
    idx_by_date = {d: i for i, d in enumerate(unique_dates)}

    pbar = tqdm(total=len(df), desc="Percentile", leave=False)
    vals = []
    for _, row in df.iterrows():
        d = row["__DATE__"]
        cur = idx_by_date[d]
        start = max(0, cur - lookback_days)
        back_dates = unique_dates[start:cur]
        if not back_dates or pd.isna(row["H2_abs"]):
            vals.append(pd.NA)
        else:
            pool = df[df["__DATE__"].isin(back_dates)]["H2_abs"].dropna()
            vals.append((pool <= row["H2_abs"]).sum() / len(pool) if len(pool) else pd.NA)
        pbar.update(1)
    pbar.close()

    df["Percentile"] = vals
    df.drop(columns="__DATE__", inplace=True)
    return df

def load_existing() -> pd.DataFrame:
    p = Path(TARGET_PKL)
    if p.exists():
        df = pd.read_pickle(p)
        df = ensure_datetime(df, ["TRADEDATE", "LSTTRADE"])
        return df
    return pd.DataFrame()

def save_all(df: pd.DataFrame):
    df = df.sort_values("TRADEDATE").drop_duplicates(subset=["TRADEDATE"], keep="last")
    df = df.reset_index(drop=True)
    Path(TARGET_PKL).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(TARGET_PKL)
    logger.info(f"🎯 Данные сохранены в {TARGET_PKL}")

# =======================
# ОСНОВНОЙ ПРОЦЕСС
# =======================

def main():
    start = datetime.now()
    logger.info("===== Запуск minute_prepare_pkl.py (без SECID) =====")

    df_all = load_existing()

    source_files = sorted(glob.glob(str(Path(SOURCE_DIR) / SOURCE_MASK)))
    if not source_files:
        logger.warning(f"Не найдены файлы по маске: {Path(SOURCE_DIR) / SOURCE_MASK}")

    for src_file in source_files:
        logger.info(f"Обработка базы: {src_file}")
        with sqlite3.connect(src_file) as conn_src:
            # Загружаем только нужные столбцы и БЕЗ SECID
            df_src = pd.read_sql(
                "SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME, LSTTRADE FROM Futures",
                conn_src,
                parse_dates=["TRADEDATE", "LSTTRADE"],
            )

        df_src = ensure_datetime(df_src, ["TRADEDATE", "LSTTRADE"])

        # Фильтр новых строк: в исходнике TRADEDATE уникален, поэтому достаточно проверки наличия
        if not df_all.empty:
            existing_dates = set(df_all["TRADEDATE"].astype("datetime64[ns]"))
            df_src = df_src[~df_src["TRADEDATE"].isin(existing_dates)]

        if df_src.empty:
            logger.info("Нет новых баров — пропускаем.")
            continue

        logger.info(f"Вычисление H2/H2_abs для {len(df_src)} записей")
        df_src = process_H2(df_src)

        if df_all.empty:
            df_all = df_src.copy()
        else:
            df_all = pd.concat([df_all, df_src], ignore_index=True)

        # Промежуточное сохранение
        save_all(df_all)

    if df_all.empty:
        logger.info("Нет данных для расчёта Percentile — завершение.")
        logger.info("===== Готово (пусто) =====")
        return

    if "Percentile" not in df_all.columns:
        df_all["Percentile"] = pd.NA

    mask_new = df_all["Percentile"].isna()
    if not mask_new.any():
        logger.info("Новых записей без Percentile нет — пересчёт не требуется.")
        save_all(df_all)
        logger.info("===== Готово без пересчёта Percentile =====")
        return

    df_new = df_all[mask_new].copy()
    last_new_date = df_new["TRADEDATE"].max().date()
    min_ctx_date = last_new_date - timedelta(days=LOOKBACK_DAYS + 1)

    ctx = df_all[df_all["TRADEDATE"].dt.date <= last_new_date]
    ctx = ctx[ctx["TRADEDATE"].dt.date >= min_ctx_date]

    logger.info(
        f"Расчёт Percentile: окно {LOOKBACK_DAYS} дней, контекст {min_ctx_date}..{last_new_date}"
    )

    ctx = compute_percentile(ctx, LOOKBACK_DAYS)

    upd = ctx[["TRADEDATE", "Percentile"]].dropna(subset=["Percentile"])
    df_all = df_all.merge(upd, on="TRADEDATE", how="left", suffixes=("", "_new"))
    need_fill = df_all["Percentile"].isna() & df_all["Percentile_new"].notna()
    df_all.loc[need_fill, "Percentile"] = df_all.loc[need_fill, "Percentile_new"]
    if "Percentile_new" in df_all.columns:
        df_all.drop(columns=["Percentile_new"], inplace=True)

    save_all(df_all)

    end = datetime.now()
    logger.info(f"===== Завершено за {end - start} =====")


if __name__ == "__main__":
    main()
