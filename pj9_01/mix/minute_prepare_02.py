#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# minute_prepare.py
"""
Объединяет минутные данные по фьючерсу RTS из нескольких SQLite-баз и добавляет колонки.
Уникальность строк обеспечивается ключом TRADEDATE (в исходных БД поле уникально).

Рассчитывается:
- H2 — изменение цены через 2 часа после открытия, допуск ±2 минуты;
- perc_25 — 25 перцентиль H2 относительно предыдущих LOOKBACK_DAYS торговых дней;;
- perc_75 — 75 перцентиль H2 относительно предыдущих LOOKBACK_DAYS торговых дней;;
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import glob
import logging
import yaml
from collections import deque

# =======================
# НАСТРОЙКИ И ЛОГИ
# =======================

SETTINGS_FILE = Path(__file__).parent / "settings_rts.yaml"
with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)

TICKER = settings.get("ticker", "MIX")
SOURCE_DIR = Path(settings.get("path_db_dir", ""))              # Папка с исходными SQLite
SOURCE_MASK = settings["path_db_min_file"].replace("{ticker}", TICKER)  # Маска файлов
LOOKBACK_BARS = int(settings.get("lookback_bars", 8000))

TARGET_PKL = f"minutes_{TICKER}_processed_p{LOOKBACK_BARS}.pkl"
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

def process_H2_nearest(df: pd.DataFrame) -> pd.DataFrame:
    # Требуются TRADEDATE, OPEN, CLOSE
    df = df.sort_values('TRADEDATE').reset_index(drop=True)

    # Цель: время через 2 часа
    df['TRADEDATE2h'] = df['TRADEDATE'] + pd.Timedelta(hours=2)

    # Будущая таблица для поиска ближайшего бара
    future = df[['TRADEDATE', 'CLOSE']].rename(
        columns={'TRADEDATE': 'TRADEDATEfuture', 'CLOSE': 'CLOSEfuture'}
    ).sort_values('TRADEDATEfuture')

    # merge_asof: ближайшая минута, допуск ±2 минуты
    # Требование: обе стороны отсортированы по ключу слияния
    merged = pd.merge_asof(
        left=df.sort_values('TRADEDATE2h'),
        right=future,
        left_on='TRADEDATE2h',
        right_on='TRADEDATEfuture',
        direction='nearest',
        tolerance=pd.Timedelta('2min')
    )

    # Возврат в исходный порядок по TRADEDATE при желании
    merged = merged.sort_values('TRADEDATE').reset_index(drop=True)

    # Расчет метрик
    merged['H2'] = merged['CLOSEfuture'] - merged['OPEN']
    # merged['H2_abs'] = merged['H2'].abs()

    # Опционально: удалить служебные
    merged.drop(columns=['TRADEDATEfuture'], inplace=True)

    return merged[['TRADEDATE', 'OPEN', 'CLOSE', 'H2']]


def add_percentile_prev_trading_bars(df: pd.DataFrame, bars_back: int = 8000) -> pd.DataFrame:
    """
    Добавляет колонки:
      - perc_25: 25-й перцентиль H2 по предыдущим bars_back барам (исключая текущий бар)
      - perc_75: 75-й перцентиль H2 по предыдущим bars_back барам (исключая текущий бар)

    Требования:
      - В df должны быть колонки ['TRADEDATE', 'H2'].
      - df должен быть отсортирован по 'TRADEDATE' по возрастанию.

    Параметры:
      - bars_back: размер окна по количеству баров назад.

    Возвращает:
      - Исходный df с добавленными колонками 'perc_25' и 'perc_75'.
    """
    if 'H2' not in df.columns:
        raise ValueError("Ожидается колонка 'H2' для расчёта перцентилей.")

    # Гарантируем сортировку по времени
    df = df.sort_values('TRADEDATE').reset_index(drop=True)

    # Чтобы не включать текущий бар в статистику — сдвигаем ряд на 1
    h2_prev = df['H2'].shift(1)

    # Категорично: используем rolling по фиксированному числу баров (а не по времени),
    # так как минута — кратные интервалы, и нужна точная длина истории.
    # min_periods можно выставить на разумный порог, например 50, чтобы избежать шумных оценок в самом начале.
    window = h2_prev.rolling(window=bars_back, min_periods=max(1, min(50, bars_back)))

    # Быстрые перцентили из скользящего окна
    df['perc_25'] = window.quantile(0.1)
    df['perc_75'] = window.quantile(0.9)

    return df

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
    logger.info("===== Запуск minute_prepare.py =====")

    df_all = pd.DataFrame()

    source_files = sorted(glob.glob(str(Path(SOURCE_DIR) / SOURCE_MASK)))
    if not source_files:
        logger.warning(f"Не найдены файлы по маске: {Path(SOURCE_DIR) / SOURCE_MASK}")

    for src_file in source_files:
        logger.info(f"Обработка базы: {src_file}")
        with sqlite3.connect(src_file) as conn_src:
            # Загружаем только нужные столбцы и БЕЗ SECID
            df_src = pd.read_sql(
                "SELECT TRADEDATE, OPEN, CLOSE FROM Futures",
                conn_src,
                parse_dates=["TRADEDATE"],
            )

        df_src = ensure_datetime(df_src, ["TRADEDATE"])

        # Фильтр новых строк: в исходнике TRADEDATE уникален, поэтому достаточно проверки наличия
        if not df_all.empty:
            existing_dates = set(df_all["TRADEDATE"].astype("datetime64[ns]"))
            df_src = df_src[~df_src["TRADEDATE"].isin(existing_dates)]

        if df_src.empty:
            logger.info("Нет новых баров — пропускаем.")
            continue

        if df_all.empty:
            df_all = df_src.copy()
        else:
            df_all = pd.concat([df_all, df_src], ignore_index=True)

    if df_all.empty:
        logger.info("Нет данных для расчёта — завершение.")
        logger.info("===== Готово (пусто) =====")
        return

    df_all = df_all.sort_values("TRADEDATE").reset_index(drop=True)

    logger.info(f"Вычисление H2 для {len(df_all)} записей")
    df_all = process_H2_nearest(df_all)
    print("Доля NA в H2:", df_all["H2"].isna().mean())
    print(df_all["H2"].describe())

    logger.info(f"Вычисление perc_25 и perc_75 для {len(df_all)} записей")
    df_all = df_all.sort_values("TRADEDATE").reset_index(drop=True)
    df_all = add_percentile_prev_trading_bars(df_all, bars_back=LOOKBACK_BARS)

    # print("Доля NA в perc_25:", df_all["perc_25"].isna().mean())
    # print(df_all["perc_25"].describe())
    # print("Доля NA в perc_75:", df_all["perc_75"].isna().mean())
    # print(df_all["perc_75"].describe())

    end = datetime.now()
    logger.info(f"===== Завершено за {end - start} =====")

    # Настройки для отображения широкого df pandas
    pd.options.display.width = 1200
    pd.options.display.max_colwidth = 100
    pd.options.display.max_columns = 100
    print(df_all)

    save_all(df_all)


if __name__ == "__main__":
    main()