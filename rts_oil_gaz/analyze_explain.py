from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# === путь к explain-файлу ===
EXPLAIN_PATH = Path("explain_topk_all.pkl")  # ← поменяй имя

# === загрузка ===
with open(EXPLAIN_PATH, "rb") as f:
    explain = pickle.load(f)

# with pd.option_context(
#         "display.width", 1000,
#         "display.max_columns", 10,
#         "display.max_colwidth", 120
# ):
#     print("Исходный датафрейм:")
#     print(explain)
#     print("Всего строк:", len(explain))

rows = []

# explain: { k -> [ { trade_date, best_j_date, score, pairs, body_cur, body_prev } ] }
for k, records in explain.items():
    for rec in records:
        trade_date = rec["trade_date"]
        pl = rec["body_cur"]
        win = np.sign(rec["body_cur"]) == np.sign(rec["body_prev"])

        for p in rec["pairs"]:
            # text_a — заголовки текущего дня
            text = p["text_a"]

            # иногда text — это несколько строк (чанк)
            # разобьём на отдельные заголовки
            headlines = [h.strip() for h in text.split("\n") if h.strip()]

            for h in headlines:
                rows.append({
                    "k_days": k,
                    "trade_date": trade_date,
                    "headline": h,
                    "pl": pl,
                    "win": win,
                    "similarity": p["similarity"]
                })

df = pd.DataFrame(rows)
with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 100
):
    print("Исходный датафрейм:")
    print(df.head(20))
    print("Всего строк:", len(df))

def normalize_headline(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'")
    s = s.strip()
    return s

df["headline_norm"] = df["headline"].map(normalize_headline)

agg = (
    df
    .groupby("headline_norm")
    .agg(
        trades=("pl", "count"),
        wins=("win", "sum"),
        losses=("win", lambda x: (~x).sum()),
        avg_pl=("pl", "mean"),
        sum_pl=("pl", "sum"),
        avg_sim=("similarity", "mean")
    )
    .reset_index()
)

# фильтр по минимуму сделок
agg = agg[agg["trades"] >= 10]

# сортировка по суммарному P/L
agg = agg.sort_values("sum_pl", ascending=False)

with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 100
):
    print("\nАгрегация: какие заголовки тащат")
    print(agg.head(20))

white_list = agg[
    (agg["trades"] >= 20) &
    (agg["avg_pl"] > 0) &
    (agg["wins"] / agg["trades"] > 0.6)
]

with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 100
):
    print("\nWHITE-LIST кандидаты")
    print(white_list.head(20))

black_list = agg[
    (agg["trades"] >= 20) &
    (agg["avg_pl"] < 0) &
    (agg["wins"] / agg["trades"] < 0.4)
]

with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 100
):
    print("\nBLACK-LIST кандидаты")
    print(black_list.head(20))

by_k = (
    df
    .groupby(["headline_norm", "k_days"])
    .agg(
        trades=("pl", "count"),
        avg_pl=("pl", "mean"),
        sum_pl=("pl", "sum")
    )
    .reset_index()
)
with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 100
):
    print("\nРазрез по k_days")
    print(by_k.sort_values("sum_pl", ascending=False).head(30))
