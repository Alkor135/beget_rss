#!/usr/bin/env python3
"""
Симуляционная торговля на основе предсказаний из XLSX-файлов.

Скрипт обрабатывает серию Excel-файлов с именами в формате 'YYYY-MM-DD.xlsx',
расположенных в указанной директории. Файлы обрабатываются в строгом хронологическом порядке.

Логика работы:
1. Начиная с даты START_DATE, для каждого файла:
   - Загружается текущий DataFrame (df).
   - Проверяется, что последняя строка df соответствует дате файла.
   - Загружается предыдущий файл (df_prev) из списка.
   - Определяется колонка в df_prev с максимальным значением в последней строке (исключая 'test_date').
   - Извлекается число из имени этой колонки (например, из 'model_13' → 13).
   - В текущем df берутся последние два значения этой колонки, вычисляется разница (прибыль/убыток).
2. Результаты (дата, номер модели, P/L, кумулятивный P/L) сохраняются в итоговый DataFrame.
3. Результат записывается в Excel-файл.
4. Строится график:
   - Столбцы: номер модели (prev_max) по датам.
   - Линия: кумулятивный Profit/Loss.

Требования к данным:
- Каждый XLSX-файл должен содержать колонку 'test_date' (дата в формате YYYY-MM-DD).
- Последняя строка файла должна иметь test_date, совпадающую с именем файла.
- Колонки с предсказаниями должны быть числовыми, предпочтительно с именами вида 'model_N'.

Результат:
- result.xlsx — таблица с результатами симуляции.
- plot_prev_max_cumulative.png — визуализация выбора моделей и кумулятивной прибыли.
"""

from pathlib import Path
import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

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

    # Извлечение числа из prev_max_col
    number = int(re.search(r'\d+', prev_max_col).group())

    # === Запись результата в df_rez ===
    new_row = pd.DataFrame({
        "Дата": [file_date.strftime("%Y-%m-%d")],  # Сохраняем дату как строку без времени
        "prev_max": [number],  # Число из названия колонки с максимальным значением в последней строке
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

# === Построение графика ===
# Убедимся, что 'Дата' в формате datetime для корректной оси X
df_rez['Дата'] = pd.to_datetime(df_rez['Дата'])

# Создаём фигуру и оси
fig, ax1 = plt.subplots(figsize=(12, 6))

# Столбчатый график для prev_max
ax1.bar(df_rez['Дата'], df_rez['prev_max'], color='skyblue', label='prev_max', width=0.6)
ax1.set_xlabel('Дата')
ax1.set_ylabel('prev_max', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# Вторая ось Y для линейного графика
ax2 = ax1.twinx()
ax2.plot(df_rez['Дата'], df_rez['Cumulative Profit/Loss'], color='green', marker='o', linewidth=2,
         label='Cumulative Profit/Loss')
ax2.set_ylabel('Cumulative Profit/Loss', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Заголовок и легенда
plt.title('prev_max (столбцы) и Cumulative Profit/Loss (линия)')
fig.tight_layout()

# Легенда
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Сохранение графика в файл
plot_file = FOLDER.parent / "plot_prev_max_cumulative.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ График сохранён в: {plot_file}")

print("\nГотово.")
