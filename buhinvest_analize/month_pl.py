"""
Построение столбчатого графика доходности по месяцам из файла Buhinvest в RUR
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Читаем файл, выбираем лист "Data" и нужные колонки
file_path = r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx"

df = pd.read_excel(file_path, sheet_name="Data", usecols=["Дата", "Profit/Loss к предыдущему"])

# Убедиться, что "Дата" - это datetime
df['Дата'] = pd.to_datetime(df['Дата'])

# Преобразовать Profit/Loss в числовой формат (если там текст)
df['Profit/Loss к предыдущему'] = pd.to_numeric(df['Profit/Loss к предыдущему'], errors='coerce')

# Замена NaN на 0 в 'Profit/Loss к предыдущему'
df["Profit/Loss к предыдущему"] = df["Profit/Loss к предыдущему"].fillna(0)

# Удалить строки с NaT в Дата
df = df.dropna(subset=['Дата'])

# Сортировка по дате
df = df.sort_values('Дата')

# Выводим информацию
print("\nДатафрейм:")
print(df)
print("\nТипы данных:")
print(df.dtypes)

# Добавим столбец месяца и агрегируем
monthly = df.copy()
monthly["Месяц"] = monthly["Дата"].dt.to_period("M")
pl_by_month = monthly.groupby("Месяц", as_index=False)["Profit/Loss к предыдущему"].sum()

# Для удобной оси X переведем период в Timestamp (конец месяца)
pl_by_month["Месяц_dt"] = pl_by_month["Месяц"].dt.to_timestamp()
# Переименование колонки
pl_by_month = pl_by_month.rename(columns={"Profit/Loss к предыдущему": "Profit/Loss"})

print("\nProfit/Loss по месяцам:")
print(pl_by_month[['Месяц', 'Profit/Loss']])

plt.figure(figsize=(10, 5))
ax = plt.gca()
ax.bar(pl_by_month["Месяц_dt"], pl_by_month["Profit/Loss"], width=20)

# Формат оси X: ГГГГ-ММ
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45, ha="right")

plt.title("Сумма Profit/Loss по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Сумма Profit/Loss")

# Подписи числовых значений над столбцами
for x, y in zip(pl_by_month["Месяц_dt"], pl_by_month["Profit/Loss"]):
    ax.text(x, y, f"{y:,.0f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(r"pl_by_month.png", dpi=200, bbox_inches="tight")
# plt.show()  # можно оставить, но в неинтерактивной среде окно не появится
