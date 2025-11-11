from pathlib import Path
from datetime import datetime, date
import re
import logging

# --- Конфигурация ---
# Торгуемые инструменты и количество
ticker_close = 'MXZ5'  # Инструмент для закрытия позиции
quantity_close = '1'
ticker_open = 'MXZ5'  # Инструмент для открытия новой позиции
quantity_open = '1'
ticker_lc = 'mix'  # Название инструмента в нижнем регистре для путей

# Пути к файлам
predict_path = Path(f"C:/Users/Alkor/gd/predict_ai/{ticker_lc}_investing_ollama")
log_path = predict_path / "log"
trade_path = Path(r"C:\QUIK_VTB_2025_ЕБС\algotrade")
trade_filepath = trade_path / "input.tri"

# Создание необходимых директорий
trade_path.mkdir(parents=True, exist_ok=True)
log_path.mkdir(parents=True, exist_ok=True)

# Имя файла прогноза на текущую дату
today = date.today()
current_filename = today.strftime("%Y-%m-%d") + ".txt"
current_filepath = predict_path / current_filename

# Настройка логгирования
log_file = log_path / f'trade_{ticker_lc}_tri.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Вспомогательные функции ---
def get_direction(filepath):
    """
    Извлекает предсказание (up/down) из указанного файла.
    Проверяет несколько кодировок для корректного чтения.
    """
    encodings = ['utf-8', 'cp1251']
    for encoding in encodings:
        try:
            with filepath.open('r', encoding=encoding) as f:
                for line in f:
                    if "Предсказанное направление:" in line:
                        direction = line.split(":", 1)[1].strip().lower()
                        if direction in ['up', 'down']:
                            return direction
            return None
        except UnicodeDecodeError:
            continue
    logger.error(f"Не удалось прочитать файл {filepath} с кодировками {encodings}.")
    return None

def get_next_trans_id(trade_filepath):
    """
    Определяет следующий TRANS_ID на основе максимального значения в файле.
    """
    trans_id = 1
    if trade_filepath.exists():
        try:
            with trade_filepath.open('r', encoding='cp1251') as f:
                content = f.read()
                trans_ids = re.findall(r'TRANS_ID=(\d+);', content)
                if trans_ids:
                    trans_id = max(int(tid) for tid in trans_ids if tid.isdigit()) + 1
        except (UnicodeDecodeError, ValueError) as e:
            logger.error(f"Ошибка при чтении TRANS_ID из {trade_filepath}: {e}")
    return trans_id

# --- Основная логика ---
# Проверка наличия файла прогноза на сегодня
if not current_filepath.exists() or current_filepath.stat().st_size == 0:
    logger.warning(f"Файл {current_filepath} не существует или пуст.")
    exit(1)

# Сбор и сортировка всех .txt файлов по дате
files = []  # Список имен всех файлов предсказаний
for filepath in predict_path.glob("*.txt"):
    try:
        file_date = datetime.strptime(filepath.stem, "%Y-%m-%d").date()
        files.append((file_date, filepath.name))
    except ValueError:
        continue

files.sort(key=lambda x: x[0], reverse=True)  # Сортировка списка имен всех файлов с предсказаниями

# Поиск текущего и предыдущего файла
current_date = today  # Текущая дата
prev_filename = None  # Имя файла с предыдущим предсказанием
for i, (file_date, filename) in enumerate(files):
    if file_date == current_date:
        if i + 1 < len(files):
            prev_filename = files[i + 1][1]
        break
logger.info(f"Предыдущий файл предсказаний: {prev_filename}")

if prev_filename is None:
    logger.info("Предыдущий файл не найден.")
    exit(1)

prev_filepath = predict_path / prev_filename

# Получение направлений из текущего и предыдущего файлов
current_predict = get_direction(current_filepath)
prev_predict = get_direction(prev_filepath)

if current_predict is None or prev_predict is None:
    logger.warning("Не удалось найти предсказанное направление в одном или обоих файлах.")
    exit()

# --- Формирование сигнала ---
trans_id = get_next_trans_id(trade_filepath)
expiry_date = today.strftime("%Y%m%d")
trade_direction = None
trade_content = None

def create_trade_block(tr_id, ticker, action, quantity):
    """Формирует блок транзакции в зависимости от направления и инструмента."""
    return (
        f'TRANS_ID={tr_id};'
        f'CLASSCODE=SPBFUT;'
        f'ACTION=Ввод заявки;'
        f'Торговый счет=SPBFUT192yc;'
        f'К/П={action};'
        f'Тип=Рыночная;'
        f'Класс=SPBFUT;'
        f'Инструмент={ticker};'
        f'Цена=0;'
        f'Количество={quantity};'
        f'Условие исполнения=Поставить в очередь;'
        f'Комментарий=SPBFUT192yc//TRI;'
        f'Переносить заявку=Нет;'
        f'Дата экспирации={expiry_date};'
        f'Код внешнего пользователя=;\n'
    )

# --- Логика выбора направления ---
# Проверка на совпадение инструментов (тикеры одинаковые)
if ticker_close == ticker_open:
    # Условия для переворота позиций
    if current_predict == 'down' and prev_predict == 'up':
        trade_direction = 'SELL'
        trade_content = (
            create_trade_block(trans_id, ticker_close, 'Продажа', quantity_close) +
            create_trade_block(trans_id+1, ticker_open, 'Продажа', quantity_open)
        )
    elif current_predict == 'up' and prev_predict == 'down':
        trade_direction = 'BUY'
        trade_content = (
            create_trade_block(trans_id, ticker_close, 'Покупка', quantity_close) +
            create_trade_block(trans_id+1, ticker_open, 'Покупка', quantity_open)
        )
# Условие ролловера (тикеры разные)
elif ticker_close != ticker_open:
    # Условия для переворота позиций
    if current_predict == 'down' and prev_predict == 'up':
        trade_direction = 'SELL'
        trade_content = (
                create_trade_block(trans_id, ticker_close, 'Продажа', quantity_close) +
                create_trade_block(trans_id+1, ticker_open, 'Продажа', quantity_open)
        )
    elif current_predict == 'up' and prev_predict == 'down':
        trade_direction = 'BUY'
        trade_content = (
                create_trade_block(trans_id, ticker_close, 'Покупка', quantity_close) +
                create_trade_block(trans_id+1, ticker_open, 'Покупка', quantity_open)
        )
    # Условия для переоткрытия позиций по новому тикеру
    elif current_predict == 'down' and prev_predict == 'down':
        trade_direction = 'SELL'
        trade_content = (
                create_trade_block(trans_id, ticker_close, 'Продажа', quantity_close) +
                create_trade_block(trans_id+1, ticker_open, 'Продажа', quantity_open)
        )
    elif current_predict == 'up' and prev_predict == 'up':
        trade_direction = 'BUY'
        trade_content = (
                create_trade_block(trans_id, ticker_close, 'Покупка', quantity_close) +
                create_trade_block(trans_id+1, ticker_open, 'Покупка', quantity_open)
        )

# --- Запись результата ---
if trade_content:
    with trade_filepath.open('a', encoding='cp1251') as f:
        f.write(trade_content)
    logger.info(f'{prev_predict=}, {current_predict=}')
    logger.info(f"Добавлена транзакция {trade_direction} с TRANS_ID={trans_id} в файл {trade_filepath}.")
else:
    logger.info(
        f"На {today} условия для сигналов BUY или SELL не выполнены. "
        f"{prev_predict=}, {current_predict=}")