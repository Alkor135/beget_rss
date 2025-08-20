"""
Скрипт для скачивания файлов с сервера Beget по SSH с использованием библиотеки Paramiko.
Этот скрипт подключается к серверу по SSH, скачивает необходимые файлы и сохраняет их в локальные директории.
Логи процесса скачивания сохраняются в отдельный файл с ротацией по времени (каждый день).
Используется библиотека Paramiko для работы с SSH и SFTP.
"""

import paramiko
from pathlib import Path
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pytz import timezone

# Создание папки для логов перед настройкой логирования
log_dir = Path('C:\\Users\\Alkor\\gd\\data_beget_rss\\log')
log_dir.mkdir(parents=True, exist_ok=True)
# Определение файла логов
log_file = Path('C:\\Users\\Alkor\\gd\\data_beget_rss\\log\\download_log.log')

# Настройка логирования с ротацией по времени
log_handler = TimedRotatingFileHandler(
    log_file,
    when='midnight',  # Новый файл каждый день в полночь
    interval=1,
    backupCount=3,    # Хранить логи за 3 дней
    encoding='utf-8'  # Указываем кодировку UTF-8
)
# log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
log_handler.setFormatter(logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    style='%'
))
log_handler.converter = lambda *args: datetime.now(timezone('Europe/Moscow')).timetuple()
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(log_handler)

def download_files():
    """Скачивание файлов с сервера Beget"""
    try:
        # Параметры подключения к серверу
        hostname = '109.172.46.10'  # IP-адрес вашего сервера
        username = 'root'           # Имя пользователя
        key_path = 'C:\\Users\\Alkor\\.ssh\\id_rsa'  # Путь к приватному SSH-ключу

        # Локальные папки для сохранения
        local_db_dir = Path('C:\\Users\\Alkor\\gd\\data_beget_rss')  # Для баз данных
        local_log_dir = Path('C:\\Users\\Alkor\\gd\\data_beget_rss\\log')  # Для логов
        local_db_dir.mkdir(parents=True, exist_ok=True)
        local_log_dir.mkdir(parents=True, exist_ok=True)

        # Удаляем все файлы в local_db_dir и всех её вложенных папках, кроме download_log.log
        for old_file in local_db_dir.rglob("*"):
            if old_file.is_file() and old_file != log_file:  # Проверяем, что это файл и не download_log.log
                logging.info(f'Удаление файла: {old_file}')
                old_file.unlink()

        # Файлы для скачивания с сервера
        remote_files = [
            ('/home/user/rss_scraper/db_data/rss_news_investing.db', local_db_dir),
            ('/home/user/rss_scraper/db_data/rss_news_investing_2025_06.db', local_db_dir),
            ('/home/user/rss_scraper/db_data/rss_news_investing_2025_07.db', local_db_dir),
            ('/home/user/rss_scraper/db_data/rss_news_investing_2025_08.db', local_db_dir),
            # ('/home/user/rss_scraper/db_data/RTS_day_rss_2025.db', local_db_dir),
            # ('/home/user/rss_scraper/db_data/MIX_day_rss_2025.db', local_db_dir),
            ('/home/user/rss_scraper/log/rss_scraper.log', local_log_dir),
            ('/home/user/rss_scraper/log/rss_scraper_month.log', local_log_dir)
            # ('/home/user/rss_scraper/log/rts_quote_download_to_db.log', local_log_dir),
            # ('/home/user/rss_scraper/log/mix_quote_download_to_db.log', local_log_dir)
        ]

        # Настройка SSH-клиента
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, username=username, key_filename=key_path)

        # Создание SFTP-клиента
        with ssh_client.open_sftp() as sftp:
            for remote_file, local_dir in remote_files:
                try:
                    sftp.stat(remote_file)  # Проверка существования файла на сервере
                    local_file = local_dir / Path(remote_file).name
                    sftp.get(remote_file, str(local_file))
                    logging.info(f"Скачан файл: {remote_file} -> {local_file}")
                except FileNotFoundError:
                    logging.warning(f"Файл {remote_file} не найден на сервере")
                except Exception as e:
                    logging.error(f"Ошибка при скачивании {remote_file}: {e}")

        ssh_client.close()
        logging.info("Процесс скачивания завершен")

    except Exception as e:
        logging.error(f"Ошибка при скачивании: {e}")

if __name__ == '__main__':
    logging.info(f"\nЗапуск процесса скачивания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    download_files()
