"""
Полный анализ фьючерсного счёта:
- Доход в рублях и процентах (TWR, IRR/MWR)
- Абсолютная и относительная просадка
- Фактор восстановления (Recovery Factor)
- Годовая доходность (CAGR)
- Экстраполяция годовой доходности по всему периоду
- Графики капитала и месячной доходности
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from numpy_financial import irr

# === Путь к Excel ===
file_path = Path(r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx")
data_df = pd.read_excel(file_path, sheet_name="Data")

# --- Очистка и подготовка ---
data_df = data_df[["Дата", "Вводы", "Выводы", "Всего на счетах", "Общ. приыбль Руб."]].dropna(how="all")
data_df["Дата"] = pd.to_datetime(data_df["Дата"])
data_df = data_df.sort_values("Дата").reset_index(drop=True)

# --- Капитал и потоки ---
data_df["Equity"] = data_df["Всего на счетах"]
data_df["Вводы"] = data_df["Вводы"].fillna(0)
data_df["Выводы"] = data_df["Выводы"].fillna(0)
data_df["NetFlow"] = data_df["Вводы"] - data_df["Выводы"]

# --- Доходность (TWR) ---
data_df["DailyReturn"] = data_df["Equity"].pct_change().fillna(0)
data_df["YearMonth"] = data_df["Дата"].dt.to_period("M")

monthly = data_df.groupby("YearMonth").agg(
    StartEquity=("Equity", "first"),
    EndEquity=("Equity", "last"),
    Flows=("NetFlow", "sum"),
)
monthly["Profit_RUB"] = monthly["EndEquity"] - monthly["StartEquity"] - monthly["Flows"]
monthly["TWR"] = (monthly["EndEquity"] / (monthly["StartEquity"] + monthly["Flows"])) - 1
monthly["AnnReturn"] = (1 + monthly["TWR"]) ** 12 - 1

# --- Просадки ---
data_df["Peak"] = data_df["Equity"].cummax()
data_df["Drawdown"] = (data_df["Equity"] - data_df["Peak"]) / data_df["Peak"]
abs_drawdown = data_df["Equity"] - data_df["Peak"]

max_dd_abs = abs_drawdown.min()
max_dd_rel = data_df["Drawdown"].min()
recovery_factor = (data_df["Equity"].iloc[-1] - data_df["Equity"].min()) / abs(max_dd_abs) if max_dd_abs != 0 else np.nan

# --- IRR (реальная доходность инвестора) ---
flows = []
for i in range(len(data_df)):
    f = -float(data_df.loc[i, "Вводы"] or 0) + float(data_df.loc[i, "Выводы"] or 0)
    if i == len(data_df) - 1:
        f += float(data_df.loc[i, "Всего на счетах"] or 0)
    flows.append(f)
flows = [x for x in flows if np.isfinite(x)]

irr_daily = np.nan
irr_annual = np.nan
if len(flows) > 1 and any(abs(x) > 1e-6 for x in flows):
    try:
        irr_daily = irr(flows)
        if np.isfinite(irr_daily):
            irr_annual = (1 + irr_daily) ** 365 - 1
    except Exception as e:
        print(f"[!] Не удалось вычислить IRR: {e}")

# --- Совокупные показатели ---
start_total = monthly["StartEquity"].iloc[0]
end_total = monthly["EndEquity"].iloc[-1]
total_twr = (end_total / start_total) - 1
total_days = (data_df["Дата"].iloc[-1] - data_df["Дата"].iloc[0]).days
total_years = total_days / 365
total_cagr = (end_total / start_total) ** (1 / total_years) - 1 if total_years > 0 else np.nan

# --- Экстраполяция годовой доходности по всему периоду ---
# (если текущая доходность сохраняется, какой будет годовой результат)
annualized_total_return = (1 + total_twr) ** (365 / total_days) - 1 if total_days > 0 else np.nan

# --- Реальный доход ---
total_invested = data_df["Вводы"].sum() - data_df["Выводы"].sum()
real_profit = data_df["Всего на счетах"].iloc[-1] - total_invested

# === Вывод ===
print("=== Месячные результаты ===")
print(monthly[["StartEquity", "EndEquity", "Flows", "Profit_RUB", "TWR", "AnnReturn"]])

for idx, row in monthly.iterrows():
    ym = str(idx)
    twr = row["TWR"]
    profit_rub = row["Profit_RUB"]
    flows = row["Flows"]
    comment = f"\nМесяц {ym}: "
    comment += f"{'прибыль' if twr > 0 else 'убыток'} {twr:.2%} ({profit_rub:,.0f} руб.). "
    if flows > 0:
        comment += f"Ввод средств: {flows:,.0f} руб. "
    elif flows < 0:
        comment += f"Вывод средств: {abs(flows):,.0f} руб. "
    comment += f"Экстраполяция годовой доходности: {row['AnnReturn']:.2%}."
    print(comment)

print("\n=== Совокупные результаты ===")
print(f"Общая доходность (TWR): {total_twr:.2%}")
print(f"CAGR (годовая): {total_cagr:.2%}")
print(f"Экстраполяция годовой доходности по всем данным: {annualized_total_return:.2%}")
print(f"Макс. просадка: {max_dd_rel:.2%} ({max_dd_abs:,.0f} руб.)")
print(f"Фактор восстановления: {recovery_factor:.2f}")
print(f"Реальная доходность инвестора (IRR годовая): {irr_annual:.2%}" if np.isfinite(irr_annual) else "IRR не удалось вычислить")
print(f"Фактическая прибыль: {real_profit:,.0f} руб. при вложениях {total_invested:,.0f} руб.")

# === Графики ===
plt.figure(figsize=(10,5))
plt.plot(data_df["Дата"], data_df["Equity"], label="Equity Curve", color="blue")
plt.fill_between(data_df["Дата"], data_df["Peak"], data_df["Equity"], color="red", alpha=0.2)
plt.title("Кривая капитала и просадки")
plt.xlabel("Дата")
plt.ylabel("Сумма на счетах (руб.)")
plt.legend()
plt.grid(True)
plt.savefig("equity_curve.png", dpi=200)

plt.figure(figsize=(8,4))
monthly["TWR"].plot(kind="bar", color=["green" if x > 0 else "red" for x in monthly["TWR"]])
plt.title("Месячная доходность (TWR)")
plt.xlabel("Месяц")
plt.ylabel("Доходность")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("monthly_returns.png", dpi=200)

print("\n✅ Анализ завершён. Сохранены графики: equity_curve.png и monthly_returns.png")
