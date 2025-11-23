#!/usr/bin/env python3
"""
Обработка xlsx файлов, имя которых — дата в формате YYYY-MM-DD.xlsx.
Файлы сортируются в строгой календарной последовательности.
Начало обработки задаётся переменной START_DATE.
"""

from pathlib import Path
from time import strftime

import pandas as pd
from datetime import datetime

# === НАСТРОЙКИ ===
FOLDER = Path(__file__).parent / "xlsx_files"   # путь к каталогу с файлами
START_DATE = "2025-07-30"                       # начиная с этой даты (включительно)

def col_name_max_val_end_str(df):
    """Поиск имени колонки с максимальным значением в последней строке."""
    # Исключаем 'test_date' и все нечисловые колонки, если нужно только среди числовых
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if 'test_date' in numeric_cols:
        numeric_cols.remove('test_date')

    # Получаем значения последней строки для оставшихся колонок
    last_row = df_prev.iloc[-1][numeric_cols]

    # Находим колонку с максимальным значением
    max_col = last_row.idxmax()
    max_value = last_row.max()

    print(
        f"✅ Колонка с максимальным значением в последней строке (кроме 'test_date'):"
        f" {max_col} = {max_value}")
    return max_col, max_value

# Преобразуем в datetime
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").date()

# === Сканирование и отбор файлов ===
xlsx_files = []

for file in FOLDER.iterdir():
    if file.suffix.lower() != ".xlsx":
        continue

    name = file.stem  # часть без .xlsx (ожидаем YYYY-MM-DD)

    try:
        file_date = datetime.strptime(name, "%Y-%m-%d").date()
    except ValueError:
        # имя не является датой — пропускаем
        continue

    xlsx_files.append((file_date, file))

# сортируем по дате
xlsx_files.sort(key=lambda x: x[0])
print(xlsx_files)
print(f"Найдено {len(xlsx_files)} файлов для обработки.")

# === Обработка файлов ===
df_rez = pd.DataFrame()  # Датафрейм с результатом
for idx, (file_date, filepath) in enumerate(xlsx_files):
    if file_date < start_dt:
        continue

    print(f"\n⏳ Обрабатываю файл [{idx}]: {filepath.name}")

    # Загружаем текущий файл
    df = pd.read_excel(filepath)
    df['test_date'] = pd.to_datetime(df['test_date']).dt.date  # сразу в тип date

    # Проверяем, что последняя строка df имеет test_date == file_date
    if df.empty or df.iloc[-1]['test_date'] != file_date:
        print(
            f"❌ Последняя строка test_date не совпадает с ожидаемой датой файла:"
            f" {file_date}. Пропускаем...")
        del df
        continue

    print(
        f"Размер текущего DF (idx={idx}) (file_date={file_date.strftime('%Y-%m-%d')}): {df.shape}")

    # Загружаем предыдущий файл (с индексом idx-1), если он существует и прошёл фильтр по дате
    df_prev = None
    if idx > 0:
        prev_file_date, prev_filepath = xlsx_files[idx - 1]
        df_prev = pd.read_excel(prev_filepath)
        print(
            f"Размер предыдущ DF (idx={idx - 1}) "
            f"(prev_file_date={prev_file_date.strftime('%Y-%m-%d')}): {df_prev.shape}")
    else:
        print("Это первый файл в списке — предыдущего файла нет.")

    # >>>>> тут твоя логика обработки df и df_prev <<<<<<
    prev_max_col, prev_max_value = col_name_max_val_end_str(df_prev)

    # Проверяем, существует ли колонка prev_max_col в df
    if prev_max_col not in df.columns:
        print(f"❌ Колонка {prev_max_col} не найдена в текущем датафрейме. Пропускаем...")
        del df
        del df_prev
        continue

    # Получаем значения последней и предпоследней строк для колонки prev_max_col
    last_value = df.iloc[-1][prev_max_col]
    second_last_value = df.iloc[-2][prev_max_col]

    # Вычисляем разницу
    diff = last_value - second_last_value

    print(f"✅ Разница в колонке {prev_max_col}: {last_value} - {second_last_value} = {diff}")

    # === Запись результата в df_rez ===
    new_row = pd.DataFrame({
        "Дата": [file_date.strftime("%Y-%m-%d")],  # Сохраняем дату как строку без времени
        "Profit/Loss": [diff]
    })
    df_rez = pd.concat([df_rez, new_row], ignore_index=True)

    # === Расчёт кумулятивной суммы ===
    df_rez["Cumulative Profit/Loss"] = df_rez["Profit/Loss"].cumsum()

    # Очистка временных DataFrames
    del df
    if df_prev is not None:
        del df_prev

print(df_rez)

# === Сохранение результата в Excel ===
result_file = FOLDER.parent / "result.xlsx"
df_rez.to_excel(result_file, index=False)
print(f"\n✅ Результат сохранён в: {result_file}")

print("\nГотово.")
