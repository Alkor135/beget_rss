#!/usr/bin/env python3
"""
Обработка xlsx файлов, имя которых — дата в формате YYYY-MM-DD.xlsx.
Файлы сортируются в строгой календарной последовательности.
Начало обработки задаётся переменной START_DATE.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

# === НАСТРОЙКИ ===
FOLDER = Path(__file__).parent / "xlsx_files"   # путь к каталогу с файлами
START_DATE = "2025-07-31"                       # начиная с этой даты (включительно)

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

    if file_date >= start_dt:
        xlsx_files.append((file_date, file))

# сортируем по дате
xlsx_files.sort(key=lambda x: x[0])

print(f"Найдено {len(xlsx_files)} файлов для обработки.")

# === Обработка файлов ===
for file_date, filepath in xlsx_files:
    print(f"\n⏳ Обрабатываю файл: {filepath.name}")

    # Загружаем во временный df
    df = pd.read_excel(filepath)

    # >>>>> тут твоя логика обработки df <<<<<<
    print(f"Размер DF: {df.shape}")

    # пример — просто показать первые строки
    # print(df.head())

    # Очистка временного DataFrame
    del df

print("\nГотово.")
