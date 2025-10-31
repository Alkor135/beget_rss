#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# minute_prepare.py
"""
Объединяет минутные данные по фьючерсу RTS из нескольких SQLite-баз и сохраняет
результат в PKL. Уникальность строк обеспечивается ключом TRADEDATE (в исходных БД поле уникально).

Рассчитывается:
- H2 — изменение цены через 2 часа после открытия;
- H2_abs — абсолютное значение H2;
- Percentile — процентиль H2_abs относительно предыдущих LOOKBACK_DAYS торговых дней.
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
    merged['H2_abs'] = merged['H2'].abs()

    # Опционально: удалить служебные
    merged.drop(columns=['TRADEDATEfuture'], inplace=True)

    return merged[['TRADEDATE', 'OPEN', 'CLOSE', 'H2', 'H2_abs']]


def add_percentile_prev_trading_days(df: pd.DataFrame, days_back: int = 10) -> pd.DataFrame:
    # Требуются колонки: TRADEDATE (datetime64[ns]), H2_abs (float)
    out = df.copy()
    # Определяем торговый день; при ночных сессиях внедрите свою функцию нормализации даты
    out['TradeDay'] = out['TRADEDATE'].dt.date

    # Соберём словарь: торговый день -> индексы строк и вектор H2_abs
    day_to_idx = {}
    day_to_vals = {}
    for day, g in out.groupby('TradeDay', sort=True):
        idx = g.index.to_numpy()
        vals = g['H2_abs'].to_numpy()
        day_to_idx[day] = idx
        day_to_vals[day] = vals

    # Отсортированный список всех торговых дней
    all_days = sorted(day_to_idx.keys())

    # Перебираем дни по порядку и считаем перцентили для строк каждого дня
    percentile = np.full(len(out), np.nan, dtype=float)

    # Для ускорения будем поддерживать скользящее окно «значений предыдущих дней»
    from collections import deque
    window_vals = deque()  # очереди по дням
    window_len = 0

    for i, day in enumerate(all_days):
        # Сдвигаем окно так, чтобы в нём были до days_back предыдущих дней
        # Сначала добавим все предыдущие дни, если идём по одному - окно уже содержит прошлые
        # Обновим окно: гарантируем, что в нём не более days_back дней
        while len(window_vals) > days_back:
            removed = window_vals.popleft()
            window_len -= len(removed)

        # Соберём плоский массив значений из окна
        if window_len > 0:
            flat = np.concatenate(list(window_vals)) if len(window_vals) > 1 else window_vals[0]
            # Отсортируем для ранжирования
            flat_sorted = np.sort(flat)
        else:
            flat_sorted = np.array([], dtype=float)

        # Рассчитываем перцентиль для всех строк текущего дня на основании «прошлого окна»
        cur_idx = day_to_idx[day]
        cur_vals = out.loc[cur_idx, 'H2_abs'].to_numpy()

        if flat_sorted.size == 0:
            # Недостаточно истории — по условию вернуть NaN (или можно 0.5/нейтрально — оставим NaN)
            percentile[cur_idx] = np.nan
        else:
            # Позиционное ранжирование: P = (кол-во <= x) / N, без интерполяции
            # Можно заменить на интерполяцию между соседями при желании
            ranks = np.searchsorted(flat_sorted, cur_vals, side='right')
            percentile[cur_idx] = ranks / flat_sorted.size

        # После расчёта добавляем текущий день в окно для следующих дней
        window_vals.append(day_to_vals[day])
        window_len += len(day_to_vals[day])

        # Ограничиваем длину окна по дням (не по записям)
        while len(window_vals) > days_back:
            removed = window_vals.popleft()
            window_len -= len(removed)

    out['Percentile'] = percentile
    return out

# def add_percentile_prev_trading_days(
#     df: pd.DataFrame,
#     days_back: int = 10,
#     min_obs: int = 2000,      # минимальный размер окна в наблюдениях
#     time_bucket: str | None = None  # например 'H' чтобы сравнивать только с тем же часом
# ) -> pd.DataFrame:
#     """
#     Добавляет колонку 'Percentile' как ранговый перцентиль H2_abs относительно
#     минут из предыдущих торговых дней. Использует:
#       - учёт связей: P = (count(< x) + 0.5*count(= x)) / N
#       - сглаживание ранга: (rank - 0.5)/N для робастности
#       - минимальное число наблюдений в окне (min_obs), при его нехватке окно расширяется вглубь до days_back
#       - опциональный выбор подвыборки по времени суток (time_bucket), например по часу ('H')
#     """
#     out = df.copy()
#     assert 'TRADEDATE' in out and 'H2_abs' in out, "Нужны TRADEDATE и H2_abs"
#     out = out.sort_values('TRADEDATE').reset_index(drop=True)
#     out['TradeDay'] = out['TRADEDATE'].dt.date
#
#     if time_bucket is not None:
#         out['Bucket'] = out['TRADEDATE'].dt.to_period(time_bucket).astype(str)
#     else:
#         out['Bucket'] = 'ALL'
#
#     # Индексы и значения по дню и бакету
#     day_bucket_to_idx = {}
#     day_bucket_to_vals = {}
#     for (day, buck), g in out.groupby(['TradeDay', 'Bucket'], sort=True):
#         day_bucket_to_idx[(day, buck)] = g.index.to_numpy()
#         day_bucket_to_vals[(day, buck)] = g['H2_abs'].to_numpy()
#
#     all_days = sorted(out['TradeDay'].unique())
#     percentile = np.full(len(out), np.nan, dtype=float)
#
#     # Для каждого бакета ведём своё окно предыдущих дней
#     windows = {buck: deque() for buck in out['Bucket'].unique()}
#     windows_len = {buck: 0 for buck in out['Bucket'].unique()}
#
#     for i, day in enumerate(all_days):
#         # Подготовим «плоские окна» по каждому бакету на момент начала дня
#         flat_cache = {}
#
#         for buck in out['Bucket'].unique():
#             # Собрать текущий плоский массив окна
#             if windows_len[buck] > 0:
#                 flat = np.concatenate(list(windows[buck])) if len(windows[buck]) > 1 else windows[buck][0]
#             else:
#                 flat = np.array([], dtype=float)
#
#             # Если наблюдений меньше минимума, попробуем «растянуть» окно глубже до days_back дней назад
#             if flat.size < min_obs:
#                 # Формируем список предыдущих дней для подкачки
#                 j = 1
#                 while flat.size < min_obs and j <= days_back and (i - j) >= 0:
#                     prev_day = all_days[i - j]
#                     key = (prev_day, buck)
#                     if key in day_bucket_to_vals:
#                         windows[buck].appendleft(day_bucket_to_vals[key])
#                         windows_len[buck] += len(day_bucket_to_vals[key])
#                         flat = np.concatenate(list(windows[buck])) if len(windows[buck]) > 1 else windows[buck][0]
#                     j += 1
#
#             # Ограничиваем окно по числу дней не более days_back
#             while len(windows[buck]) > days_back:
#                 removed = windows[buck].pop()  # удаляем самые старые справа
#                 windows_len[buck] -= len(removed)
#                 if windows_len[buck] < 0:
#                     windows_len[buck] = 0
#
#             flat_cache[buck] = np.sort(flat) if flat.size else flat
#
#         # Рассчёт перцентилей для текущего дня по каждому бакету
#         for buck in out['Bucket'].unique():
#             key = (day, buck)
#             if key not in day_bucket_to_idx:
#                 continue
#             idx = day_bucket_to_idx[key]
#             vals = out.loc[idx, 'H2_abs'].to_numpy()
#             ref = flat_cache[buck]
#
#             if ref.size == 0:
#                 percentile[idx] = np.nan
#             else:
#                 # Учёт связей: P = (n_lt + 0.5*n_eq) / N
#                 # Альтернатива: сглаженный ранг (rank - 0.5)/N через searchsorted('right')
#                 n = ref.size
#                 # Для векторизации: для каждого vals считаем n_lt и n_eq
#                 # n_lt = позиция 'left', n_le = позиция 'right'; n_eq = n_le - n_lt
#                 pos_left = np.searchsorted(ref, vals, side='left')
#                 pos_right = np.searchsorted(ref, vals, side='right')
#                 n_lt = pos_left
#                 n_eq = pos_right - pos_left
#                 percentile[idx] = (n_lt + 0.5 * n_eq) / n
#
#         # По завершении дня добавляем день в окна каждого бакета
#         for buck in out['Bucket'].unique():
#             key = (day, buck)
#             if key in day_bucket_to_vals:
#                 windows[buck].append(day_bucket_to_vals[key])  # добавляем «сегодня» как самый новый
#                 windows_len[buck] += len(day_bucket_to_vals[key])
#             # Ограничение по числу дней
#             while len(windows[buck]) > days_back:
#                 removed = windows[buck].popleft()
#                 windows_len[buck] -= len(removed)
#
#     out['Percentile'] = percentile
#     return out

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

    logger.info(f"Вычисление H2/H2_abs для {len(df_all)} записей")
    df_all = process_H2_nearest(df_all)
    print("Доля NA в H2_abs:", df_all["H2_abs"].isna().mean())

    df_all = add_percentile_prev_trading_days(df_all, days_back=LOOKBACK_DAYS)
    print("Доля NA в Percentile:", df_all["Percentile"].isna().mean())
    print(df_all['Percentile'].describe())  # count, mean, std, min, quartiles,
    # Квантили
    print(df_all['Percentile'].quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    # Бины по 10% (децили)
    bins = pd.interval_range(0, 1, periods=10)
    cats = pd.cut(df_all['Percentile'], bins, include_lowest=True)
    print(cats.value_counts().sort_index())

    end = datetime.now()
    logger.info(f"===== Завершено за {end - start} =====")

    # Настройки для отображения широкого df pandas
    pd.options.display.width = 1200
    pd.options.display.max_colwidth = 100
    pd.options.display.max_columns = 100
    print(df_all)


if __name__ == "__main__":
    main()