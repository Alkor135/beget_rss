"""
Скрипт для чтения базы данных котировок и новостей с сервера, формирования markdown-файлов с заголовками новостей.
Сохраняет не более 30 последних интервалов в формате markdown с метаданными.
Использует базу данных котировок с дневными свечами, сформированными из минутных данных с 21:00 МСК.
Обрабатывает последние три файла БД новостей (по месяцам) на сервере.
"""

import pandas as pd
from pathlib import Path
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import paramiko
from sshtunnel import SSHTunnelForwarder
import re

# Параметры подключения к серверу
SSH_HOST = "109.172.46.10"  # IP-адрес сервера
SSH_PORT = 22
SSH_USERNAME = "root"  # Имя пользователя для SSH
SSH_PASSWORD = None  # Пароль (или используйте SSH_KEY_PATH)
# Путь к SSH-ключу, если используется (например, '/home/user/.ssh/id_rsa')
SSH_KEY_PATH = "C:\\Users\\Alkor\\.ssh\\id_rsa"
REMOTE_DB_DIR = "/home/user/rss_scraper/db_data"  # Путь к директории с базами данных на сервере

def msk_to_gmt(dt_str: str) -> str:
    """
    Преобразует строку даты-времени из МСК в GMT (ISO-формат).
    """
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=ZoneInfo("Europe/Moscow"))
    dt_gmt = dt.astimezone(ZoneInfo("Etc/GMT"))
    return dt_gmt.strftime("%Y-%m-%d %H:%M:%S")

def read_db_quote(db_path_quote: Path) -> pd.DataFrame:
    """
    Читает таблицу Futures из базы данных котировок и возвращает DataFrame.
    """
    with sqlite3.connect(db_path_quote) as conn:
        return pd.read_sql_query("SELECT * FROM Futures", conn)

def get_latest_db_files_ssh(num_files: int = 3) -> list[str]:
    """
    Находит последние num_files файлов БД новостей на сервере, сортируя по году и месяцу в имени (descending).
    Фильтрует только файлы формата rss_news_investing_YYYY_MM.db.
    """
    files = []
    try:
        # Подключение к серверу через SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if SSH_KEY_PATH:
            ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, key_filename=SSH_KEY_PATH)
        else:
            ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, password=SSH_PASSWORD)

        # Выполняем команду ls для получения списка файлов
        stdin, stdout, stderr = ssh.exec_command(f"ls -1 {REMOTE_DB_DIR}/rss_news_investing_*_*.db")
        file_list = stdout.read().decode().splitlines()

        # Фильтруем и сортируем файлы
        for f in file_list:
            fname = f.strip()
            match = re.match(r".*rss_news_investing_(\d{4})_(\d{2})\.db", fname)
            if match:
                year, month = map(int, match.groups())
                files.append((fname, year, month))

        ssh.close()

        # Сортировка по (год, месяц) descending
        files.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [f[0] for f in files[:num_files]]
    except Exception as e:
        print(f"Ошибка при получении списка файлов с сервера: {e}")
        return []

def read_db_news_multiple_ssh(db_paths: list[str], date_max: str, date_min: str) -> pd.DataFrame:
    """
    Читает новости из нескольких баз данных на сервере за указанный период времени через SSH-туннель.
    """
    df_list = []
    for db_path in db_paths:
        try:
            # Настройка SSH-туннеля
            with SSHTunnelForwarder(
                (SSH_HOST, SSH_PORT),
                ssh_username=SSH_USERNAME,
                ssh_password=SSH_PASSWORD,
                ssh_pkey=SSH_KEY_PATH,
                remote_bind_address=('localhost', 0),  # SQLite не требует сетевого порта, но туннель нужен для доступа к файлу
                allow_agent=SSH_KEY_PATH is not None
            ) as tunnel:
                # Подключение к базе данных
                conn = sqlite3.connect(db_path)  # Прямой доступ к файлу через туннель
                query = "SELECT * FROM news WHERE date > ? AND date < ?"
                df = pd.read_sql_query(query, conn, params=[date_min, date_max])
                conn.close()
                df_list.append(df)
        except Exception as e:
            print(f"Ошибка при чтении базы данных {db_path}: {e}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

def save_titles_to_markdown(
        df_news: pd.DataFrame,
        file_path: Path,
        next_bar: str,
        date_min: str,
        date_max: str
    ) -> None:
    """
    Сохраняет заголовки новостей в markdown-файл с метаданными.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"---\nnext_bar: {next_bar}\ndate_min: {date_min}\ndate_max: {date_max}\n---\n\n")
        for _, row in df_news.iterrows():
            title = row['title']
            file.write(f"- {title}\n")

def main(path_db_quote: Path, md_news_dir: Path, num_dbs: int = 3) -> None:
    """
    Основная функция: читает котировки локально и новости с сервера,
    удаляет старые markdown-файлы, формирует и сохраняет не более 30 markdown-файлов
    с новостями и метаданными за самые последние даты.
    """
    # Получаем последние файлы БД новостей с сервера
    db_paths = get_latest_db_files_ssh(num_files=num_dbs)
    if len(db_paths) < num_dbs:
        print(f"Предупреждение: Найдено только {len(db_paths)} файлов БД, ожидалось {num_dbs}")

    print("Используемые файлы БД:", db_paths)

    # Удаляем все старые markdown-файлы в директории
    for old_file in md_news_dir.glob("*.md"):
        if old_file.is_file():
            old_file.unlink()

    # Читаем базу данных котировок и формируем DataFrame
    df = read_db_quote(path_db_quote)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df.sort_values(by='TRADEDATE', inplace=True)
    df = df.tail(31)  # Ограничиваем до 31 строки, чтобы получить 30 интервалов
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['bar'] = df.apply(lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1)
    df['next_bar'] = df['bar'].shift(-1)

    for i in range(len(df) - 1, 0, -1):
        row1 = df.iloc[i]
        row2 = df.iloc[i - 1]

        file_name = f"{row1['TRADEDATE']}.md"
        date_max = f"{row1['TRADEDATE']} 21:00:00"
        date_min = f"{row2['TRADEDATE']} 21:00:00"
        date_max_gmt = msk_to_gmt(date_max)
        date_min_gmt = msk_to_gmt(date_min)

        print(f"{file_name}. Новости за период: {date_min} - {date_max}")
        df_news = read_db_news_multiple_ssh(db_paths, date_max_gmt, date_min_gmt)
        if len(df_news) == 0:
            break

        save_titles_to_markdown(
            df_news, Path(fr'{md_news_dir}/{file_name}'),
            row1['next_bar'], date_min_gmt, date_max_gmt
        )

if __name__ == '__main__':
    ticker = 'RTS'
    path_db_quote = Path(fr'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db')
    md_news_dir = Path('c:/Users/Alkor/gd/news_rss_md_rts_21-00_month')

    # Создаем директорию для сохранения markdown-файлов, если она не существует
    Path(md_news_dir).mkdir(parents=True, exist_ok=True)

    if not path_db_quote.exists():
        print(f"Ошибка: Файл базы данных котировок не найден. {path_db_quote}")
        exit()

    main(path_db_quote, md_news_dir)