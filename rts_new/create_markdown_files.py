import pandas as pd
from pathlib import Path
import sqlite3
import logging
import yaml

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
ticker_lc = ticker.lower()
num_mds = settings['num_mds']  # Количество последних интервалов для сохранения в markdown файлы
num_dbs = settings['num_dbs']  # Количество последних файлов БД новостей для обработки
# Время с которого начинается поиск новостей за предыдущую сессию в БД
time_start = settings['time_start']
# Время, которым заканчивается поиск новостей за текущую сессию в БД
time_end = settings['time_end']

# Директория с БД дневных свечей с 21:00 предыдущей сессии до 21:00 даты свечи.
path_db_day = settings['path_db_day']
# Директория с файлами БД новостей по месяцам
db_news_dir = settings['db_news_dir']
# Директория для сохранения markdown-файлов с новостями с 21:00 МСК предыдущей торговой сессии
md_path = Path(settings['md_path'])  # Путь к markdown-файлам
(Path(__file__).parent / 'log').mkdir(parents=True, exist_ok=True)  # Создание папки для логов, если ее нет
log_file = Path(__file__).parent / 'log' / 'create_markdown_files.txt'  # Файл лога
