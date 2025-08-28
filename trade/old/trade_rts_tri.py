from pathlib import Path
from datetime import datetime, date
import re

# Параметры
# Путь к папке с файлами прогнозов
predict_path = Path(r"C:\Users\Alkor\gd\predict_ai\rts_investing_ollama")
# Путь к папке с файлами trade
trade_path = Path(r"C:\QUIK_VTB_2025_ЕБС\algotrade")
trade_path.mkdir(parents=True, exist_ok=True)  # Создание папки, если не существует
today = date.today()  # Получение текущей даты
# Формирование имени файла для текущей даты (формат: ГГГГ-ММ-ДД.txt)
current_filename = today.strftime("%Y-%m-%d") + ".txt"
# Полный путь к файлу текущей даты
current_filepath = predict_path / current_filename

# Проверка существования файла за текущую дату
if not current_filepath.exists():
    print(f"Файл за текущую дату {current_filename} не найден.")
    exit()

# Сбор всех .txt файлов и их дат
files = []
for filepath in predict_path.glob("*.txt"):
    try:
        # Извлечение даты из имени файла
        file_date = datetime.strptime(filepath.stem, "%Y-%m-%d").date()
        files.append((file_date, filepath.name))
    except ValueError:
        continue  # Пропуск файлов с некорректным форматом имени

# Сортировка файлов по дате в порядке убывания
files.sort(key=lambda x: x[0], reverse=True)

# Поиск файла за текущую дату и предыдущего файла
current_date = today
prev_filename = None
for i, (file_date, filename) in enumerate(files):
    if file_date == current_date:
        # Если найден файл за текущую дату, берем следующий (предыдущий по дате)
        if i + 1 < len(files):
            prev_filename = files[i + 1][1]
        break

# Проверка наличия предыдущего файла
if prev_filename is None:
    print("Предыдущий файл не найден.")
    exit()

# Полный путь к предыдущему файлу
prev_filepath = predict_path / prev_filename

# Функция для извлечения направления из файла
def get_direction(filepath):
    # Попытка открыть файл с разными кодировками
    encodings = ['utf-8', 'cp1251']  # Список возможных кодировок
    for encoding in encodings:
        try:
            with filepath.open('r', encoding=encoding) as f:
                for line in f:
                    if "Предсказанное направление:" in line:
                        # Извлечение направления (up или down) и приведение к нижнему регистру
                        direction = line.split(":", 1)[1].strip().lower()
                        if direction in ['up', 'down']:
                            return direction
            return None
        except UnicodeDecodeError:
            continue  # Пробуем следующую кодировку
    print(f"Не удалось прочитать файл {filepath} с кодировками {encodings}.")
    return None

# Получение направлений из текущего и предыдущего файлов
current_dir = get_direction(current_filepath)
prev_dir = get_direction(prev_filepath)

# Проверка наличия направлений в обоих файлах
if current_dir is None or prev_dir is None:
    print("Не удалось найти предсказанное направление в одном или обоих файлах.")
    exit()

# Функция для получения следующего TRANS_ID
def get_next_trans_id(trade_filepath):
    trans_id = 1  # Значение по умолчанию
    if trade_filepath.exists():
        try:
            with trade_filepath.open('r', encoding='cp1251') as f:
                content = f.read()
                # Поиск всех значений TRANS_ID
                trans_ids = re.findall(r'TRANS_ID=(\d+);', content)
                if trans_ids:
                    # Преобразование в числа и поиск максимума
                    trans_id = max(int(tid) for tid in trans_ids) + 1
        except UnicodeDecodeError:
            print(f"Не удалось прочитать файл {trade_filepath} для определения TRANS_ID.")
    return trans_id

# Проверка условий для записи торгового сигнала
trade_content = None
# Формат текущей даты для поля Дата экспирации (ГГГГММДД)
expiry_date = today.strftime("%Y%m%d")
trade_filepath = trade_path / 'input.tri'
trans_id = get_next_trans_id(trade_filepath)
if current_dir == 'down' and prev_dir == 'up':
    trade_content = (
        f'TRANS_ID={trans_id};CLASSCODE=SPBFUT;ACTION=Ввод заявки;Торговый счет=SPBFUT192yc;'
        f'К/П=Продажа;Тип=Рыночная;Класс=SPBFUT;Инструмент=RIU5;Цена=0;Количество=2;'
        f'Условие исполнения=Поставить в очередь;Комментарий=SPBFUT192yc//TRI;'
        f'Переносить заявку=Нет;Дата экспирации={expiry_date};Код внешнего пользователя=;\n')
elif current_dir == 'up' and prev_dir == 'down':
    trade_content = (
        f'TRANS_ID={trans_id};CLASSCODE=SPBFUT;ACTION=Ввод заявки;Торговый счет=SPBFUT192yc;'
        f'К/П=Покупка;Тип=Рыночная;Класс=SPBFUT;Инструмент=RIU5;Цена=0;Количество=2;'
        f'Условие исполнения=Поставить в очередь;Комментарий=SPBFUT192yc//TRI;'
        f'Переносить заявку=Нет;Дата экспирации={expiry_date};Код внешнего пользователя=;\n')

# Запись результата в файл trade, если условия выполнены
if trade_content:
    # Получение следующего TRANS_ID
    trans_id = get_next_trans_id(trade_filepath)
    # Форматирование строки с актуальным TRANS_ID
    trade_content = trade_content.format(trans_id=trans_id)
    # Добавление записи в файл trade в кодировке cp1251
    with trade_filepath.open('a', encoding='cp1251') as f:
        f.write(trade_content)
    print(f'{current_dir=}, {prev_dir=}')
    print(f"Добавлен сигнал с TRANS_ID={trans_id} в файл {trade_filepath}.")
else:
    print("Условия для сигналов BUY или SELL не выполнены.")