import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# # Для IDE без GUI (например, PyCharm) раскомментируйте следующую строку:
# import matplotlib
# matplotlib.use("TkAgg")

# === Загрузка данных из Excel ===
file_path = Path(r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx")
data_df = pd.read_excel(file_path, sheet_name="Data")

# --- Очистка и подготовка ---
data_df = data_df[["Дата", "Вводы", "Выводы", "Всего на счетах", "Общ. приыбль Руб."]].dropna(how="all")
data_df["Дата"] = pd.to_datetime(data_df["Дата"])
data_df = data_df.sort_values("Дата").reset_index(drop=True)

# --- Ежедневная доходность (TWR) ---
data_df["Equity"] = data_df["Всего на счетах"]

# Убираем кэшфлоу (вводы/выводы) для расчета TWR
cashflow = data_df[["Дата", "Вводы", "Выводы"]].fillna(0)
data_df["NetFlow"] = cashflow["Вводы"] - cashflow["Выводы"]

# Скорректированный капитал (без вводов/выводов)
data_df["Adj_Equity"] = data_df["Equity"] - data_df["NetFlow"].cumsum()

# Доходность по дням
data_df["DailyReturn"] = data_df["Equity"].pct_change().fillna(0)

# --- Группировка по месяцам ---
data_df["YearMonth"] = data_df["Дата"].dt.to_period("M")

monthly = data_df.groupby("YearMonth").agg(
    StartEquity=("Equity", "first"),
    EndEquity=("Equity", "last"),
    Flows=("NetFlow", "sum"),
)

# TWR по месяцам
monthly["TWR"] = (monthly["EndEquity"] / (monthly["StartEquity"] + monthly["Flows"])) - 1

# Годовая доходность (CAGR)
periods_per_year = 12
monthly["AnnReturn"] = (1 + monthly["TWR"]) ** periods_per_year - 1

# --- Итоговый отчет ---
print("=== Месячные результаты ===")
print(monthly[["StartEquity", "EndEquity", "Flows", "TWR", "AnnReturn"]])

# --- Комментарии по метрикам ---
for idx, row in monthly.iterrows():
    ym = str(idx)
    start, end, flows, twr, ann = row["StartEquity"], row["EndEquity"], row["Flows"], row["TWR"], row["AnnReturn"]
    comment = []
    if twr > 0:
        comment.append(f"Месяц {ym}: положительная доходность {twr:.2%}.")
    else:
        comment.append(f"Месяц {ym}: убыток {twr:.2%}.")
    if abs(flows) > 0:
        if flows > 0:
            comment.append(f"Ввод средств составил {flows:,.2f} руб.")
        else:
            comment.append(f"Вывод средств составил {abs(flows):,.2f} руб.")
    comment.append(f"Годовая доходность (экстраполяция): {ann:.2%}.")
    print(" ".join(comment))

# --- Совокупные показатели ---
start_total = monthly["StartEquity"].iloc[0]
end_total = monthly["EndEquity"].iloc[-1]
total_twr = (end_total / start_total) - 1
total_years = (data_df["Дата"].iloc[-1] - data_df["Дата"].iloc[0]).days / 365
if total_years > 0:
    total_cagr = (end_total / start_total) ** (1 / total_years) - 1
else:
    total_cagr = np.nan

print("\n=== Совокупные результаты ===")
print(f"Общая доходность (TWR): {total_twr:.2%}")
if not np.isnan(total_cagr):
    print(f"CAGR (годовая доходность за весь период): {total_cagr:.2%}")

# --- График капитала ---
plt.figure(figsize=(10,5))
plt.plot(data_df["Дата"], data_df["Equity"], label="Equity Curve")
plt.title("Кривая капитала")
plt.xlabel("Дата")
plt.ylabel("Сумма на счетах")
plt.legend()
plt.grid(True)
plt.savefig("equity_curve.png", dpi=200)

# --- График месячных доходностей ---
plt.figure(figsize=(8,4))
monthly["TWR"].plot(kind="bar", color=["green" if x > 0 else "red" for x in monthly["TWR"]])
plt.title("Месячная доходность (TWR)")
plt.xlabel("Месяц")
plt.ylabel("Доходность")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("monthly_returns.png", dpi=200)
# plt.show()  # Раскомментируйте для интерактивного режима
