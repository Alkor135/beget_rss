"""
Динамический бектест:
- Для каждой свечи (начиная с 31-й) выбираем N_PAST ∈ [2..10],
  который показал наилучшую суммарную доходность на последних 20 свечах симуляции.
- Прогноз: направление свечи с наибольшим телом (|CLOSE - OPEN|) из N_PAST прошлых.
- Сохраняем результаты и сводку в Excel + вставляем график (PNG).
Выбор фьючерса зависит от параметра ticker в файле settings.yaml.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
import logging
import sys
import yaml

# ---------------------------
# Параметры
# ---------------------------
# SETTINGS_FILE = Path(__file__).parent / "settings_rts.yaml"
SETTINGS_FILE = Path(__file__).parent / "settings_mix.yaml"
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings['ticker']
ticker_lc = ticker.lower()
DB_PATH = Path(fr"C:\Users\Alkor\gd\data_quote_db\{ticker}_days.sqlite")
OUTPUT_XLSX = Path(__file__).parent / f"{ticker_lc}_strategy_backtest_dynamic_strongest.xlsx"
PNG_PATH = OUTPUT_XLSX.with_suffix(".png")

N_PAST_RANGE = range(2, 11)  # (2..10)
WINDOW = 20                  # окно для оценки (20)
START_INDEX = 30             # старт с 31-й свечи
RANDOM_SEED = 42             # оставим для симуляции детерминированность

# ---------------------------
# Логирование
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Вспомогательные функции
# ---------------------------
def find_table_with_columns(conn: sqlite3.Connection, required_cols):
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
    if not db_path.exists():
        raise FileNotFoundError(f"Файл БД не найден: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        table = find_table_with_columns(conn, ["TRADEDATE", "OPEN", "CLOSE"])
        if not table:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tbls = [r[0] for r in cur.fetchall()]
            cur.close()
            raise RuntimeError(f"Не найдена таблица с колонками TRADEDATE, OPEN, CLOSE. "
                               f"Доступные таблицы: {tbls}")
        logger.info(f"Используем таблицу `{table}` для чтения данных.")
        df = pd.read_sql_query(f"SELECT TRADEDATE, OPEN, CLOSE FROM '{table}' ORDER BY TRADEDATE ASC",
                               conn, parse_dates=["TRADEDATE"])
    finally:
        conn.close()

    if not pd.api.types.is_datetime64_any_dtype(df["TRADEDATE"]):
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"]).reset_index(drop=True)
    return df

def candle_direction(open_p, close_p):
    if close_p > open_p:
        return "UP"
    elif close_p < open_p:
        return "DOWN"
    else:
        return "FLAT"

def strongest_past_direction(df, i, npast):
    """Возвращает направление свечи с наибольшим телом из последних npast перед i"""
    past = df.loc[i-npast:i-1, ["OPEN", "CLOSE"]].copy()
    past["BODY"] = (past["CLOSE"] - past["OPEN"]).abs()
    idx = past["BODY"].idxmax()
    return candle_direction(past.loc[idx, "OPEN"], past.loc[idx, "CLOSE"])

# ---------------------------
# Главная логика
# ---------------------------
def main():
    logger.info("Запуск динамического бектеста (прогноз = свеча с самым большим телом)")
    try:
        df = safe_read_df(DB_PATH)
    except Exception as e:
        logger.error(f"Ошибка чтения данных из БД: {e}")
        sys.exit(1)

    if len(df) <= START_INDEX:
        logger.error(f"В БД слишком мало свечей ({len(df)}). Нужно минимум {START_INDEX+1}.")
        sys.exit(1)

    df["FACT_DIR"] = df.apply(lambda r: candle_direction(r["OPEN"], r["CLOSE"]), axis=1)

    records = []
    cum_result = 0.0

    n_total = len(df)
    logger.info(f"Всего свечей: {n_total}. Стартуем с {START_INDEX} ({df.loc[START_INDEX,'TRADEDATE']}).")

    for i in range(START_INDEX, n_total):
        # Окно для оценки
        window_start = i - WINDOW
        window_end = i

        npast_perf = {}
        for npast in N_PAST_RANGE:
            tmp_cum = 0.0
            for j in range(window_start, window_end):
                if j - npast < 0:
                    continue
                pred = strongest_past_direction(df, j, npast)
                open_p, close_p = df.loc[j, ["OPEN", "CLOSE"]]
                if pred == "UP":
                    tmp_cum += close_p - open_p
                elif pred == "DOWN":
                    tmp_cum += open_p - close_p
            npast_perf[npast] = tmp_cum

        # выбираем лучший N_PAST
        max_perf = max(npast_perf.values())
        best_candidates = [k for k, v in npast_perf.items() if v == max_perf]
        best_npast = min(best_candidates)

        # делаем прогноз на свечу i
        pred_dir = strongest_past_direction(df, i, best_npast)

        open_p, close_p = df.loc[i, ["OPEN", "CLOSE"]]
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

        if (i - START_INDEX) % 100 == 0:
            logger.info(f"Обработано {i - START_INDEX + 1} / {n_total - START_INDEX} свечей...")

    res_df = pd.DataFrame(records)
    if res_df.empty:
        logger.error("Результирующий DataFrame пуст.")
        sys.exit(1)

    total_trades = len(res_df)
    correct = res_df["CORRECT"].sum()
    accuracy = correct / total_trades * 100.0
    final_result = res_df["CUM_RESULT"].iloc[-1]
    avg_npast = res_df["N_PAST"].mean()

    # --- новые метрики ---
    avg_profit = final_result / total_trades if total_trades > 0 else 0.0

    running_max = res_df["CUM_RESULT"].cummax()
    drawdowns = running_max - res_df["CUM_RESULT"]
    max_drawdown = drawdowns.max()  # абсолютная просадка (максимальная просадка)
    rel_drawdown = max_drawdown / running_max.max() if running_max.max() != 0 else 0.0

    summary_df = pd.DataFrame({
        "Всего сделок": [total_trades],
        "Правильных прогнозов": [int(correct)],
        "Доля правильных (%)": [accuracy],
        "Итоговый результат": [final_result],
        "Средний N_PAST": [avg_npast],
        "Средняя прибыль на сделку": [avg_profit],
        "Максимальная просадка": [max_drawdown],
        "Относительная просадка": [rel_drawdown],
    })

    dist = res_df["N_PAST"].value_counts().sort_index()
    dist_df = pd.DataFrame({"N_PAST": dist.index, "Count": dist.values})

    logger.info(f"Сохраняем результаты в {OUTPUT_XLSX}")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name="Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        dist_df.to_excel(writer, sheet_name="N_PAST_Distribution", index=False)

    logger.info("Строим график кумулятивного результата")
    plt.figure(figsize=(12, 6))
    plt.plot(res_df["TRADEDATE"], res_df["CUM_RESULT"], label="CUM_RESULT")
    plt.xlabel("Дата")
    plt.ylabel("Кумулятивный результат")
    plt.title("Бектест (сильнейшая свеча): кумулятивный результат")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()

    wb = load_workbook(OUTPUT_XLSX)
    ws = wb["Summary"]
    img = OpenpyxlImage(str(PNG_PATH))
    ws.add_image(img, "G2")
    wb.save(OUTPUT_XLSX)

    logger.info("Готово. Файл сохранён: %s", OUTPUT_XLSX)
    logger.info("Сводка:\n%s", summary_df.to_string(index=False))
    logger.info("Распределение N_PAST:\n%s", dist_df.to_string(index=False))

if __name__ == "__main__":
    main()
