from pathlib import Path
from datetime import datetime, date
import re
import logging

# Параметры
ticker = 'RIU5'  # Торгуемый инструмент
ticker_lc = 'rts'
quantity = '2'

predict_path = Path(
    fr"C:\Users\Alkor\gd\predict_ai\{ticker_lc}_investing_ollama")  # Путь к папке с файлами прогнозов
log_path = Path(
    fr"C:\Users\Alkor\gd\predict_ai\{ticker_lc}_investing_ollama\log")  # Путь к папке с файлом логов
trade_path = Path(r"C:\QUIK_VTB_2025_ЕБС\algotrade")  # Путь к папке с файлами trade
trade_path.mkdir(parents=True, exist_ok=True)  # Создание папки, если не существует
trade_filepath = trade_path / 'input.tri'
today = date.today()  # Получение текущей даты
current_filename = today.strftime("%Y-%m-%d") + ".txt"  # Файл предсказания для текущей даты (формат: ГГГГ-ММ-ДД.txt)
current_filepath = predict_path / current_filename  # Полный путь к файлу предсказаний для текущей даты

# Настройка логирования
log_path.mkdir(parents=True, exist_ok=True)  # Создание папки для логов, если не существует
log_file = log_path / 'trade_rts_tri.txt'

# Конфигурация логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),  # Перезапись файла лога
        logging.StreamHandler()  # Вывод в консоль
    ]
)
logger = logging.getLogger(__name__)

# Проверка существования файла за текущую дату
if not current_filepath.exists():
    logger.error(
        f"Файл за текущую дату {current_filename} не найден в {current_filepath}. "
        f"Проверьте наличие файла.")
    exit(1)

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
    logger.info("Предыдущий файл не найден.")
    exit(1)

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
    logger.error(f"Не удалось прочитать файл {filepath} с кодировками {encodings}.")
    return None

# Получение направлений из текущего и предыдущего файлов
current_predict = get_direction(current_filepath)
prev_predict = get_direction(prev_filepath)

# Проверка наличия направлений в обоих файлах
if current_predict is None or prev_predict is None:
    logger.warning("Не удалось найти предсказанное направление в одном или обоих файлах.")
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
            logger.info(f"Не удалось прочитать файл {trade_filepath} для определения TRANS_ID.")
    return trans_id

# Проверка условий для записи торгового сигнала
trade_content = None  # Переменная для хранения строки транзакции
expiry_date = today.strftime("%Y%m%d")  # Формат текущей даты для поля Дата экспирации (ГГГГММДД)
trans_id = get_next_trans_id(trade_filepath)  # Получение следующего TRANS_ID
trade_direction = None  # Переменная для хранения направления транзакции
if current_predict == 'down' and prev_predict == 'up':
    trade_direction = 'SELL'
    trade_content = (
        f'TRANS_ID={trans_id};'
        f'CLASSCODE=SPBFUT;'
        f'ACTION=Ввод заявки;'
        f'Торговый счет=SPBFUT192yc;'
        f'К/П=Продажа;'
        f'Тип=Рыночная;'
        f'Класс=SPBFUT;'
        f'Инструмент={ticker};'
        f'Цена=0;'
        f'Количество={quantity};'
        f'Условие исполнения=Поставить в очередь;'
        f'Комментарий=SPBFUT192yc//TRI;'
        f'Переносить заявку=Нет;'
        f'Дата экспирации={expiry_date};'
        f'Код внешнего пользователя=;\n')
elif current_predict == 'up' and prev_predict == 'down':
    trade_direction = 'BUY'
    trade_content = (
        f'TRANS_ID={trans_id};'
        f'CLASSCODE=SPBFUT;'
        f'ACTION=Ввод заявки;'
        f'Торговый счет=SPBFUT192yc;'
        f'К/П=Покупка;'
        f'Тип=Рыночная;'
        f'Класс=SPBFUT;'
        f'Инструмент={ticker};'
        f'Цена=0;'
        f'Количество={quantity};'
        f'Условие исполнения=Поставить в очередь;'
        f'Комментарий=SPBFUT192yc//TRI;'
        f'Переносить заявку=Нет;'
        f'Дата экспирации={expiry_date};'
        f'Код внешнего пользователя=;\n')

# Запись результата в файл input.tri, если условия выполнены
if trade_content:
    # Добавление записи в файл input.tri в кодировке cp1251
    with trade_filepath.open('a', encoding='cp1251') as f:
        f.write(trade_content)
    logger.info(f'{current_predict=}, {prev_predict=}')
    logger.info(f"Добавлена транзакция {trade_direction} с TRANS_ID={trans_id} в файл {trade_filepath}.")
else:
    logger.info(
        f"На {today} условия для сигналов BUY или SELL не выполнены. "
        f"{prev_predict=}, {current_predict=}")