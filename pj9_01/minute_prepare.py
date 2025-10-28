#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# minute_prepare.py

"""
Сценарий предназначен для объединения минутных данных по фьючерсу на индекс РТС (RTS),
расположенных в нескольких SQLite-базах. Скрипт выполняет расчёт:
• H2 — изменение цены через 2 часа после открытия;
• H2_abs — абсолютное значение H2;
• Percentile — процентиль H2_abs относительно предыдущих 10 торговых дней.
Percentile теперь вычисляется только для новых записей (где он ещё не рассчитан).
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import glob
import logging
import yaml

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings_rts.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# =======================
# НАСТРОЙКИ
# =======================

TICKER = settings.get("ticker", "RTS")
SOURCE_DIR = Path(settings.get("path_db_dir", ""))   # Папка, где хранятся исходные базы SQLite
SOURCE_MASK = settings['path_db_min_file'].replace('{ticker}', TICKER)  # Маска поиска исходных файлов
LOOKBACK_DAYS = 10                                 # Количество торговых дней для расчёта Percentile
TARGET_DB = f"minutes_RTS_processed_p{LOOKBACK_DAYS}.db"  # Имя выходной базы
LOG_FILE = "minute_prepare.log"                    # Имя файла для логов

# =======================
# ЛОГИРОВАНИЕ
# =======================

# Создаём логгер
logger = logging.getLogger("minute_prepare")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Лог в консоль
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Лог в файл
fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# =======================
# ФУНКЦИИ
# =======================

def create_target_table(conn):
    """Создаёт таблицу с обработанными данными, если она ещё не существует."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS FuturesProcessed (
            TRADEDATE TEXT PRIMARY KEY UNIQUE NOT NULL,
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
    """Рассчитывает изменение цены через 2 часа (H2) и его абсолютное значение (H2_abs)."""
    # Добавляем столбец с меткой времени через 2 часа
    df["TRADEDATE_2h"] = df["TRADEDATE"] + timedelta(hours=2)

    # Формируем таблицу со сдвинутыми на 2 часа значениями CLOSE
    df_future = df[["TRADEDATE", "SECID", "CLOSE"]].rename(
        columns={"TRADEDATE": "TRADEDATE_future", "CLOSE": "CLOSE_future"}
    )

    # Объединяем текущие и будущие значения по времени
    df = df.merge(df_future, left_on=["TRADEDATE_2h", "SECID"],
                  right_on=["TRADEDATE_future", "SECID"], how="left")

    # Разница между ценой через 2 часа и ценой открытия
    df["H2"] = df["CLOSE_future"] - df["OPEN"]

    # Абсолютное изменение
    df["H2_abs"] = df["H2"].abs()

    # Удаляем временные вспомогательные колонки
    df.drop(columns=["TRADEDATE_2h", "TRADEDATE_future", "CLOSE_future"], inplace=True)

    return df


def compute_percentile(df, lookback_days):
    """
    Вычисляет процентиль для H2_abs относительно предыдущих lookback_days торговых дней.
    Процентиль выражается в диапазоне [0.0, 1.0].
    """
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    df["Percentile"] = None
    df["TRADEDATE_DATE"] = df["TRADEDATE"].dt.date

    # Список уникальных торговых дат
    unique_dates = df["TRADEDATE_DATE"].drop_duplicates().tolist()
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}

    # Прогресс-бар для визуализации процесса
    pbar = tqdm(total=len(df), desc="Вычисление Percentile")

    for idx, row in df.iterrows():
        current_date = row["TRADEDATE_DATE"]
        current_idx = date_to_idx[current_date]

        # Формируем окно lookback_days (предыдущие торговые дни)
        start_idx = max(0, current_idx - lookback_days)
        lookback_dates = unique_dates[start_idx:current_idx]

        # Если нет предыдущих дней — процентиль не вычисляем
        if not lookback_dates:
            df.at[idx, "Percentile"] = None
        else:
            pool = df[df["TRADEDATE_DATE"].isin(lookback_dates)]["H2_abs"].dropna()

            if len(pool) == 0:
                df.at[idx, "Percentile"] = None
            else:
                # Определяем долю значений, не превосходящих текущее
                rank = (pool <= row["H2_abs"]).sum()
                df.at[idx, "Percentile"] = rank / len(pool)

        pbar.update(1)

    pbar.close()
    df.drop(columns=["TRADEDATE_DATE"], inplace=True)
    return df


# =======================
# ГЛАВНАЯ ФУНКЦИЯ
# =======================

def main():
    """Основной процесс обработки всех файлов."""
    # Подключаемся к выходной базе
    with sqlite3.connect(TARGET_DB) as conn_target:
        create_target_table(conn_target)

        # Получаем список всех файлов с исходными данными
        source_files = sorted(glob.glob(str(Path(SOURCE_DIR) / SOURCE_MASK)))

        for src_file in source_files:
            logger.info(f"Обработка базы: {src_file}")

            with sqlite3.connect(src_file) as conn_src:
                # Загружаем исходные минутные данные
                df_src = pd.read_sql("SELECT * FROM Futures", conn_src,
                                     parse_dates=["TRADEDATE", "LSTTRADE"])

                # Отбираем только новые записи, которых ещё нет в целевой базе
                existing_dates = pd.read_sql(
                    "SELECT TRADEDATE FROM FuturesProcessed", conn_target
                )["TRADEDATE"]
                df_src = df_src[~df_src["TRADEDATE"].astype(str).isin(existing_dates)]

                if df_src.empty:
                    logger.info(f"Нет новых баров для обработки в {src_file}")
                    continue

                # === Расчёт изменения за 2 часа ===
                logger.info(f"Вычисление H2 / H2_abs ({len(df_src)} записей)")
                df_src = process_H2(df_src)

                # Добавляем новые записи в базу
                df_src.to_sql("FuturesProcessed", conn_target, if_exists="append", index=False)
                logger.info(f"Сохранили H2 / H2_abs в {TARGET_DB}")

                # === Новый блок: выборочный расчёт Percentile ===
                logger.info(f"Вычисление Percentile (окно {LOOKBACK_DAYS} дней)")

                # Загружаем все данные, чтобы иметь контекст для lookback
                df_all = pd.read_sql("SELECT * FROM FuturesProcessed", conn_target,
                                     parse_dates=["TRADEDATE", "LSTTRADE"])

                # Определяем новые записи без Percentile
                df_new = df_all[df_all["Percentile"].isna()].copy()

                if df_new.empty:
                    logger.info("Новых записей без Percentile не найдено — пропускаем расчёт")
                else:
                    # Определяем дату последней новой записи
                    last_new_date = df_new["TRADEDATE"].max().date()

                    # Берём контекст за последние lookback_days + 1 день
                    min_date_for_context = last_new_date - timedelta(days=LOOKBACK_DAYS + 1)
                    df_context = df_all[df_all["TRADEDATE"].dt.date <= last_new_date]
                    df_context = df_context[df_context["TRADEDATE"].dt.date >= min_date_for_context]

                    # Считаем Percentile только на этом контексте
                    df_context = compute_percentile(df_context, LOOKBACK_DAYS)

                    # Из контекста выбираем только новые строки
                    updated_rows = df_context[df_context["TRADEDATE"].isin(df_new["TRADEDATE"])]
                    logger.info(f"Обновляем {len(updated_rows)} записей с новыми Percentile")

                    # Обновляем строки напрямую через SQL UPDATE
                    with conn_target:
                        for _, row in updated_rows.iterrows():
                            conn_target.execute(
                                "UPDATE FuturesProcessed SET Percentile = ? WHERE TRADEDATE = ?",
                                (row["Percentile"], row["TRADEDATE"].strftime("%Y-%m-%d %H:%M:%S"))
                            )

                    logger.info("✅ Percentile обновлены только для новых записей")

                # Оптимизация базы после обработки
                logger.info(f"Выполняем VACUUM → {TARGET_DB}")
                conn_target.execute("VACUUM")

        logger.info("🎯 Обработка всех файлов завершена успешно")

# =======================
# ТОЧКА ВХОДА
# =======================

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info("===== Запуск process_quotes.py =====")
    main()
    end_time = datetime.now()
    logger.info(f"===== Завершено. Время исполнения: {end_time - start_time} =====")
