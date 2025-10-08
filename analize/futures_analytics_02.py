import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# === Загрузка данных ===
file_path = Path(r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx")
data_df = pd.read_excel(file_path, sheet_name="Data")

# --- Подготовка ---
data_df = data_df[["Дата", "Вводы", "Выводы", "Всего на счетах", "Общ. приыбль Руб."]].dropna(how="all")
data_df["Дата"] = pd.to_datetime(data_df["Дата"])
data_df = data_df.sort_values("Дата").reset_index(drop=True)

data_df["Equity"] = data_df["Всего на счетах"]
cashflow = data_df[["Дата", "Вводы", "Выводы"]].fillna(0)
data_df["NetFlow"] = cashflow["Вводы"] - cashflow["Выводы"]
data_df["DailyReturn"] = data_df["Equity"].pct_change().fillna(0)
data_df["YearMonth"] = data_df["Дата"].dt.to_period("M")

# === Месячная агрегация ===
monthly = data_df.groupby("YearMonth").agg(
    StartEquity=("Equity", "first"),
    EndEquity=("Equity", "last"),
    Flows=("NetFlow", "sum"),
)

# Расчет дохода в рублях и процентов
monthly["ProfitRub"] = monthly["EndEquity"] - (monthly["StartEquity"] + monthly["Flows"])
monthly["TWR"] = monthly["ProfitRub"] / (monthly["StartEquity"] + monthly["Flows"])
monthly["AnnReturn"] = (1 + monthly["TWR"]) ** 12 - 1

# === Просадка (Drawdown) ===
data_df["Peak"] = data_df["Equity"].cummax()
data_df["Drawdown"] = data_df["Equity"] - data_df["Peak"]
data_df["DrawdownPct"] = data_df["Drawdown"] / data_df["Peak"]

max_drawdown = data_df["DrawdownPct"].min()  # в долях
abs_drawdown = data_df.loc[data_df["DrawdownPct"].idxmin(), "Drawdown"]  # в рублях

# === Совокупные показатели ===
start_total = monthly["StartEquity"].iloc[0]
end_total = monthly["EndEquity"].iloc[-1]
total_twr = (end_total / start_total) - 1
total_years = (data_df["Дата"].iloc[-1] - data_df["Дата"].iloc[0]).days / 365
total_cagr = (end_total / start_total) ** (1 / total_years) - 1 if total_years > 0 else np.nan
recovery_factor = (end_total - start_total) / abs(abs_drawdown) if abs_drawdown != 0 else np.nan

# === Отчет ===
print("=== Месячные результаты ===")
print(monthly[["StartEquity", "EndEquity", "Flows", "ProfitRub", "TWR", "AnnReturn"]])

# --- Комментарии ---
for idx, row in monthly.iterrows():
    ym = str(idx)
    profit_rub = row["ProfitRub"]
    flows, twr, ann = row["Flows"], row["TWR"], row["AnnReturn"]
    comment = [f"Месяц {ym}: результат {profit_rub:,.2f} руб. ({twr:.2%})"]
    if flows != 0:
        comment.append(f"{'Ввод' if flows>0 else 'Вывод'} средств: {abs(flows):,.2f} руб.")
    comment.append(f"Экстраполяция годовой доходности: {ann:.2%}")
    print(" ".join(comment))

print("\n=== Совокупные результаты ===")
print(f"Общая доходность (TWR): {total_twr:.2%}")
if not np.isnan(total_cagr):
    print(f"CAGR (годовая доходность): {total_cagr:.2%}")

print("\n=== Просадки и устойчивость ===")
print(f"Максимальная просадка: {max_drawdown:.2%} ({abs_drawdown:,.0f} руб.)")
print(f"Фактор восстановления: {recovery_factor:.2f}")

# === Графики ===
plt.figure(figsize=(10,5))
plt.plot(data_df["Дата"], data_df["Equity"], label="Equity Curve", lw=2)
plt.fill_between(data_df["Дата"], data_df["Peak"], data_df["Equity"], color="red", alpha=0.2, label="Просадка")
plt.title("Кривая капитала с выделением просадок")
plt.xlabel("Дата")
plt.ylabel("Сумма на счетах, руб.")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("equity_curve_with_drawdown.png", dpi=200)

plt.figure(figsize=(8,4))
monthly["TWR"].plot(kind="bar", color=["green" if x>0 else "red" for x in monthly["TWR"]])
plt.title("Месячная доходность (TWR)")
plt.xlabel("Месяц")
plt.ylabel("Доходность")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("monthly_returns.png", dpi=200)
