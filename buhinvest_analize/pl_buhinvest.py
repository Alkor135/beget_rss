"""
Построение графиков доходности из файла Buhinvest в RUR.
Скрипт анализирует данные из Excel-файла Buhinvest, извлекая информацию о датах и финансовых результатах.
Он обрабатывает колонки с прибылью/убытками и общей накопленной прибылью в рублях.
На основе данных строятся два графика: столбчатый по месячной доходности и линейный по накопленной прибыли.
Для визуализации используется библиотека Matplotlib, графики сохраняются в формате PNG.
Столбцы окрашиваются в синий (прибыль) и красный (убыток) в зависимости от значения.
Скрипт автоматически обрабатывает ошибки в данных, заменяя пропуски нулями.
Результат — наглядное представление динамики торговли по дням и месяцам.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Читаем файл, выбираем лист "Data" и нужные колонки
file_path = r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx"

df = pd.read_excel(file_path, sheet_name="Data", usecols=["Дата", "Profit/Loss к предыдущему", "Общ. прибыль Руб."])

# Убедиться, что "Дата" - это datetime
df['Дата'] = pd.to_datetime(df['Дата'])

# Преобразовать столбцы в числовой формат
df['Profit/Loss к предыдущему'] = pd.to_numeric(df['Profit/Loss к предыдущему'], errors='coerce')
df['Общ. прибыль Руб.'] = pd.to_numeric(df['Общ. прибыль Руб.'], errors='coerce')

# Замена NaN на 0
df["Profit/Loss к предыдущему"] = df["Profit/Loss к предыдущему"].fillna(0)
df["Общ. прибыль Руб."] = df["Общ. прибыль Руб."].fillna(0)

# Удалить строки с NaT в Дата
df = df.dropna(subset=['Дата'])

# Сортировка по дате
df = df.sort_values('Дата')

# --- График 1: Столбчатый график по Profit/Loss по месяцам ---
monthly = df.copy()
monthly["Месяц"] = monthly["Дата"].dt.to_period("M")
pl_by_month = monthly.groupby("Месяц", as_index=False)["Profit/Loss к предыдущему"].sum()
pl_by_month["Месяц_dt"] = pl_by_month["Месяц"].dt.to_timestamp()
pl_by_month = pl_by_month.rename(columns={"Profit/Loss к предыдущему": "Profit/Loss"})

# Определяем цвета: красный для отрицательных, синий для положительных
colors = ['red' if x < 0 else 'skyblue' for x in pl_by_month["Profit/Loss"]]

plt.figure(figsize=(10, 5))
ax = plt.gca()
ax.bar(pl_by_month["Месяц_dt"], pl_by_month["Profit/Loss"], width=20, color=colors, edgecolor='black')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45, ha="right")
plt.title("Сумма Profit/Loss по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Profit/Loss (RUB)")
# Подписи на столбцах
for x, y in zip(pl_by_month["Месяц_dt"], pl_by_month["Profit/Loss"]):
    va = "top" if y < 0 else "bottom"
    ax.text(x, y, f"{y:,.0f}", ha="center", va=va, fontsize=9)
plt.tight_layout()
plt.savefig(r"pl_by_month.png", dpi=200, bbox_inches="tight")
plt.close()

# --- График 2: Линейный график по "Общ. прибыль Руб." ---
# Убираем дубликаты по дате (на случай, если есть несколько записей в один день)
cumulative = df.drop_duplicates(subset=["Дата"]).sort_values("Дата")

plt.figure(figsize=(12, 6))
plt.plot(cumulative["Дата"], cumulative["Общ. прибыль Руб."], marker='o', linestyle='-', color='green')
plt.title("Общая прибыль (накопительно) по дням")
plt.xlabel("Дата")
plt.ylabel("Общ. прибыль Руб.")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"cumulative_profit.png", dpi=200, bbox_inches="tight")
plt.close()

print("Графики сохранены:")
print("- pl_by_month.png — месячный Profit/Loss (столбцы)")
print("- cumulative_profit.png — накопительная прибыль (линия)")
