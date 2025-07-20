import paramiko
from pathlib import Path
import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Настройка логирования с ротацией по времени
log_handler = TimedRotatingFileHandler(
    'C:\\Users\\Alkor\\gd\\data_beget_rss\\download_log.log',
    when='midnight',  # Новый файл каждый день в полночь
    interval=1,
    backupCount=7,    # Хранить логи за 7 дней
    encoding='utf-8'  # Указываем кодировку UTF-8
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').setLevel(logging.INFO)
logging.getLogger('').addHandler(log_handler)

def download_files():
    """Скачивание файлов с сервера Beget"""
    try:
        # Параметры подключения к серверу
        hostname = '109.172.46.10'  # IP-адрес вашего сервера
        username = 'root'           # Имя пользователя
        key_path = 'C:\\Users\\Alkor\\.ssh\\id_rsa'  # Путь к приватному SSH-ключу

        # Локальная папка для сохранения
        local_dir = Path('C:\\Users\\Alkor\\gd\\data_beget_rss')
        local_dir.mkdir(parents=True, exist_ok=True)

        # Файлы для скачивания с сервера
        remote_files = [
            '/home/user/rss_scraper/rss_news_investing.db',
            '/home/user/rss_scraper/RTS_day_rss_2025.db',
            '/home/user/rss_scraper/rss_scraper.log',
            '/home/user/rss_scraper/futures_scraper.log'
        ]

        # Настройка SSH-клиента
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, username=username, key_filename=key_path)

        # Создание SFTP-клиента
        with ssh_client.open_sftp() as sftp:
            for remote_file in remote_files:
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
    logging.info(f"Запуск процесса скачивания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    download_files()