"""
Динамический бектест:
- Для каждой свечи (начиная с 31-й) выбираем N_PAST ∈ [2..10],
  который показал наилучшую суммарную доходность на последних 20 свечах симуляционной торговли.
- После выбора N_PAST делаем прогноз случайным выбором направления
  из последних N_PAST фактических направлений и считаем результат по OPEN->CLOSE.
- Сохраняем результаты и сводку в Excel + вставляем график (PNG) в лист "Сводка".
"""

import sqlite3
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
import logging
import sys

# ---------------------------
# Параметры (можно менять)
# ---------------------------
DB_PATH = Path(r"C:\Users\Alkor\gd\data_quote_db\RTS_days.sqlite")
OUTPUT_XLSX = Path(__file__).parent / "rts_strategy_backtest_dynamic.xlsx"
PNG_PATH = OUTPUT_XLSX.with_suffix(".png")

N_PAST_RANGE = range(2, 11)  # 2..10
WINDOW = 20                  # окно для оценки (20 предыдущих свечей)
START_INDEX = 30             # стартуем с 31-й свечи (индексация с 0)
RANDOM_SEED = 42             # фиксированное зерно для воспроизводимости

# ---------------------------
# Логирование (простой)
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Вспомогательные функции
# ---------------------------
def find_table_with_columns(conn: sqlite3.Connection, required_cols):
    """
    Ищет в sqlite-базе таблицу, содержащую все required_cols.
    Возвращает имя таблицы или None если не найдено.
    """
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    for t in tables:
        try:
            cur.execute(f"PRAGMA table_info('{t}')")
            cols = [r[1].upper() for r in cur.fetchall()]
            if all(col.upper() in cols for col in required_cols):
                return t
        except Exception:
            continue
    return None

def safe_read_df(db_path: Path):
    """
    Подключается к БД, находит подходящую таблицу (TRADEDATE, OPEN, CLOSE)
    и возвращает DataFrame с этими колонками, отсортированный по TRADEDATE ASC.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Файл БД не найден: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        table = find_table_with_columns(conn, ["TRADEDATE", "OPEN", "CLOSE"])
        if not table:
            # если не нашли, перечислим доступные таблицы для диагностики
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tbls = [r[0] for r in cur.fetchall()]
            cur.close()
            raise RuntimeError(f"Не найдена таблица с колонками TRADEDATE, OPEN, CLOSE. "
                               f"Доступные таблицы: {tbls}")
        logger.info(f"Используем таблицу `{table}` для чтения данных.")
        # читаем данные; превращаем TRADEDATE в datetime
        df = pd.read_sql_query(f"SELECT TRADEDATE, OPEN, CLOSE FROM '{table}' ORDER BY TRADEDATE ASC", conn,
                               parse_dates=["TRADEDATE"])
    finally:
        conn.close()
    # Убедимся, что TRADEDATE datetime
    if not pd.api.types.is_datetime64_any_dtype(df["TRADEDATE"]):
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"])  # выбросим строки с некорректной датой
    df = df.reset_index(drop=True)
    return df

# ---------------------------
# Главная логика
# ---------------------------
def main():
    logger.info("Запуск динамического бектеста (выбор N_PAST по 20-оконной эффективности)")
    # Читаем данные
    try:
        df = safe_read_df(DB_PATH)
    except Exception as e:
        logger.error(f"Ошибка чтения данных из БД: {e}")
        sys.exit(1)

    if len(df) <= START_INDEX:
        logger.error(f"В БД слишком мало свечей ({len(df)}). Нужны как минимум {START_INDEX+1}.")
        sys.exit(1)

    # Добавляем фактическое направление свечи (UP/DOWN/FLAT)
    def dir_of(row):
        if row["CLOSE"] > row["OPEN"]:
            return "UP"
        elif row["CLOSE"] < row["OPEN"]:
            return "DOWN"
        else:
            return "FLAT"

    df["FACT_DIR"] = df.apply(dir_of, axis=1)

    # Результирующая таблица
    records = []
    cum_result = 0.0

    # Устанавливаем глобальное зерно для воспроизводимости;
    # внутри симуляций и основного прогона используем детерминированные генераторы.
    global_seed = RANDOM_SEED

    n_total = len(df)
    logger.info(f"Всего свечей: {n_total}. Стартуем с индекса {START_INDEX} ({df.loc[START_INDEX,'TRADEDATE']}).")

    for i in range(START_INDEX, n_total):
        # Окно последних WINDOW свечей для оценки (заканчивается на i-1)
        window_start = i - WINDOW
        window_end = i  # not inclusive in range loops below

        # Собираем производительность для каждого кандидата N_PAST
        npast_perf = {}
        for npast in N_PAST_RANGE:
            # Используем отдельный RNG для каждого npast (детерминированно)
            rng = random.Random(global_seed + npast)
            tmp_cum = 0.0
            # Прогоним симуляцию по каждой свечке j в окне [window_start, i-1]
            for j in range(window_start, window_end):
                # Для этой симуляции нужна история длины npast до j (т.е. индексы j-npast .. j-1)
                if j - npast < 0:
                    continue
                past_dirs = df.loc[j - npast: j - 1, "FACT_DIR"].tolist()
                if not past_dirs:
                    continue
                pred = rng.choice(past_dirs)
                open_p = df.loc[j, "OPEN"]
                close_p = df.loc[j, "CLOSE"]
                if pred == "UP":
                    tmp_cum += (close_p - open_p)
                elif pred == "DOWN":
                    tmp_cum += (open_p - close_p)
                else:  # FLAT
                    tmp_cum += 0.0
            npast_perf[npast] = tmp_cum

        # Выбор лучшего npast — если несколько с одинаковым результатом, выбираем минимальный npast
        max_perf = max(npast_perf.values())
        best_candidates = [k for k, v in npast_perf.items() if v == max_perf]
        best_npast = min(best_candidates)

        # Теперь делаем реальную торговлю на i-й свече, используя RNG зависящий от i, чтобы быть воспроизводимым
        rng_main = random.Random(global_seed + 1000 + i)
        past_dirs_for_trade = df.loc[i - best_npast: i - 1, "FACT_DIR"].tolist()
        if not past_dirs_for_trade:
            # На всякий случай — если нет истории (должно не случаться при выбранном START_INDEX)
            pred_dir = rng_main.choice(df.loc[:i - 1, "FACT_DIR"].tolist())
        else:
            pred_dir = rng_main.choice(past_dirs_for_trade)

        open_p = df.loc[i, "OPEN"]
        close_p = df.loc[i, "CLOSE"]
        if pred_dir == "UP":
            trade_res = close_p - open_p
        elif pred_dir == "DOWN":
            trade_res = open_p - close_p
        else:
            trade_res = 0.0

        cum_result += trade_res
        fact_dir = df.loc[i, "FACT_DIR"]

        records.append({
            "TRADEDATE": df.loc[i, "TRADEDATE"],
            "INDEX": i,
            "N_PAST": best_npast,
            "PRED_DIR": pred_dir,
            "FACT_DIR": fact_dir,
            "CORRECT": pred_dir == fact_dir,
            "TRADE_RESULT": trade_res,
            "CUM_RESULT": cum_result
        })

        # Небольшой лог прогресса
        if (i - START_INDEX) % 100 == 0:
            logger.info(f"Обработано {i - START_INDEX + 1} / {n_total - START_INDEX} свечей...")

    # Результирующий DataFrame
    res_df = pd.DataFrame(records)
    if res_df.empty:
        logger.error("Результирующий DataFrame пуст — завершение.")
        sys.exit(1)

    # Сводка
    total_trades = len(res_df)
    correct = res_df["CORRECT"].sum()
    accuracy = correct / total_trades * 100.0
    final_result = res_df["CUM_RESULT"].iloc[-1]
    avg_npast = res_df["N_PAST"].mean()

    summary_df = pd.DataFrame({
        "Всего сделок": [total_trades],
        "Правильных прогнозов": [int(correct)],
        "Доля правильных (%)": [accuracy],
        "Итоговый результат": [final_result],
        "Средний N_PAST": [avg_npast]
    })

    # Частотное распределение выбранных N_PAST
    dist = res_df["N_PAST"].value_counts().sort_index()
    dist_df = pd.DataFrame({"N_PAST": dist.index, "Count": dist.values})

    # -----------------------
    # Сохраняем в Excel
    # -----------------------
    logger.info(f"Сохраняем результаты в {OUTPUT_XLSX}")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name="Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        dist_df.to_excel(writer, sheet_name="N_PAST_Distribution", index=False)

    # -----------------------
    # Строим и сохраняем график (PNG)
    # -----------------------
    logger.info("Строим график кумулятивного результата")
    plt.figure(figsize=(12, 6))
    plt.plot(res_df["TRADEDATE"], res_df["CUM_RESULT"], label="CUM_RESULT")
    plt.xlabel("Дата")
    plt.ylabel("Кумулятивный результат")
    plt.title("Динамический бектест: кумулятивный результат")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()

    # Вставляем PNG в лист Summary
    wb = load_workbook(OUTPUT_XLSX)
    ws = wb["Summary"]
    img = OpenpyxlImage(str(PNG_PATH))
    # Добавляем картинку, начиная примерно с ячейки G2 (если нужно, можно изменить)
    ws.add_image(img, "G2")
    wb.save(OUTPUT_XLSX)

    logger.info("Готово. Файл сохранён: %s", OUTPUT_XLSX)
    logger.info("Сводка:\n%s", summary_df.to_string(index=False))
    logger.info("Распределение выбранных N_PAST (по количеству выборов):\n%s", dist_df.to_string(index=False))

if __name__ == "__main__":
    main()
