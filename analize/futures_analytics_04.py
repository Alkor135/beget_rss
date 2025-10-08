import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# === Чтение данных из Excel ===
file_path = Path(r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx")
df = pd.read_excel(file_path, sheet_name="Data")

# === Переименование под стандартные имена ===
df = df.rename(columns={
    "Дата": "Date",
    "Всего на счетах": "Balance",
    "Вводы": "Deposits",
    "Выводы": "Withdrawals"
})

# Проверка наличия ключевых столбцов
required_cols = {"Date", "Balance", "Deposits", "Withdrawals"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"В файле отсутствуют нужные колонки: {required_cols - set(df.columns)}")

# === Подготовка данных ===
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df["Deposits"] = df["Deposits"].fillna(0)
df["Withdrawals"] = df["Withdrawals"].fillna(0)
df["Flow"] = df["Deposits"] - df["Withdrawals"]

# === Расчет базовых метрик ===
df["EquityChange"] = df["Balance"].diff() - df["Flow"]

# === Месячная группировка ===
df["YearMonth"] = df["Date"].dt.to_period("M")
monthly = df.groupby("YearMonth").agg(
    StartEquity=("Balance", "first"),
    EndEquity=("Balance", "last"),
    Flows=("Flow", "sum")
)
monthly["ProfitRub"] = monthly["EndEquity"] - monthly["StartEquity"] - monthly["Flows"]
monthly["TWR_month"] = (monthly["EndEquity"] - monthly["Flows"] - monthly["StartEquity"]) / monthly["StartEquity"]

# === TWR (Time-Weighted Return) ===
twr_total = np.prod(1 + monthly["TWR_month"]) - 1
num_days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
years = num_days / 365 if num_days > 0 else 1
twr_annual = (1 + twr_total) ** (1 / years) - 1

# === CAGR (Compound Annual Growth Rate) ===
cagr = (df["Balance"].iloc[-1] / df["Balance"].iloc[0]) ** (1 / years) - 1

# === Modified Dietz ===
def modified_dietz(start_value, end_value, cashflows):
    if not cashflows:
        return (end_value - start_value) / start_value
    start_date = cashflows[0][0]
    end_date = cashflows[-1][0]
    total_days = max((end_date - start_date).days, 1)
    weighted_sum = sum(cf * (1 - (d - start_date).days / total_days) for d, cf in cashflows)
    return (end_value - start_value - sum(cf for _, cf in cashflows)) / (start_value + weighted_sum)

cashflows = list(zip(df["Date"], df["Flow"]))
dietz_return = modified_dietz(df["Balance"].iloc[0], df["Balance"].iloc[-1], cashflows)
dietz_annual = (1 + dietz_return) ** (1 / years) - 1

# === XIRR ===
def xirr(cashflows):
    def npv(rate):
        return sum(cf / (1 + rate) ** ((d - cashflows[0][0]).days / 365) for d, cf in cashflows)
    rate = 0.1
    for _ in range(100):
        f = npv(rate)
        df_val = sum(-((d - cashflows[0][0]).days / 365) * cf / (1 + rate) ** (((d - cashflows[0][0]).days / 365) + 1) for d, cf in cashflows)
        if abs(df_val) < 1e-12:
            break
        rate -= f / df_val
    return rate

xirr_flows = [(d, f) for d, f in zip(df["Date"], df["Flow"])]
xirr_flows.append((df["Date"].iloc[-1], -df["Balance"].iloc[-1]))
xirr_rate = xirr(xirr_flows)

# === Сравнение методов ===
print("=== Сравнение методов доходности ===")
print(f"TWR (всего): {twr_total:.2%}")
print(f"TWR (годовых): {twr_annual:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Modified Dietz (годовых): {dietz_annual:.2%}")
print(f"XIRR: {xirr_rate:.2%}")

# === Месячная таблица ===
print("\n=== Месячные результаты ===")
print(monthly[["StartEquity", "EndEquity", "Flows", "ProfitRub", "TWR_month"]])

# === Визуализация ===
plt.figure(figsize=(10, 5))
plt.bar(monthly.index.astype(str), monthly["TWR_month"] * 100, color=["green" if x > 0 else "red" for x in monthly["TWR_month"]])
plt.title("Месячная доходность, %")
plt.ylabel("%")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Balance"], label="Капитал")
plt.title("Кривая капитала")
plt.ylabel("Баланс, ₽")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

