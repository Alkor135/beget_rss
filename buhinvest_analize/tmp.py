"""
Проверка файла xlsx и полученного датафрейма.
"""

import pandas as pd

# Читаем файл, выбираем лист "Data" и нужные колонки
file_path = r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_без_пароля.xlsx"
df = pd.read_excel(file_path, sheet_name="Data", usecols=["Дата", "Profit/Loss к предыдущему", "Общ. прибыль Руб."])

# Сначала читаем только заголовки, чтобы проверить имена столбцов
df_check = pd.read_excel(file_path, sheet_name="Data", nrows=0)
print("Все столбцы в листе 'Data':")
print(df_check.columns.tolist())
