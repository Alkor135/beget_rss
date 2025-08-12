# """
# Скрипт для синхронизации файлов с удалённого сервера на локальный ПК.
# Синхронизирует файлы .db из /home/user/rss_scraper/db_data/ в C:\Users\Alkor\gd\data_beget_rss\
# и файлы .log из /home/user/rss_scraper/log/ в C:\Users\Alkor\gd\data_beget_rss\log\.
# Удаляет локальные файлы, которых нет на сервере (аналог rsync --delete).
# Использует SSH и SFTP для передачи файлов.
# """

import paramiko
import os
from pathlib import Path
import stat
import re

# Параметры подключения и путей
SSH_HOST = "109.172.46.10"
SSH_PORT = 22
SSH_USERNAME = "root"
SSH_KEY_PATH = "C:/Users/Alkor/.ssh/id_rsa"  # Используем сырую строку
REMOTE_DB_DIR = "/home/user/rss_scraper/db_data/"  # Директория для .db файлов
REMOTE_LOG_DIR = "/home/user/rss_scraper/log/"     # Директория для .log файлов
LOCAL_DB_DIR = Path(r"C:\Users\Alkor\gd\data_beget_rss")  # Локальная директория для .db
LOCAL_LOG_DIR = Path(r"C:\Users\Alkor\gd\data_beget_rss\log")  # Локальная директория для .log

# Паттерны для включения файлов
DB_INCLUDE_PATTERNS = [r".*\.db$"]  # Для .db файлов
LOG_INCLUDE_PATTERNS = [r".*\.log$"]  # Для .log файлов


def is_included(file_path: str, patterns: list) -> bool:
    """
    Проверяет, соответствует ли файл указанным паттернам включения.
    """
    for pattern in patterns:
        if re.match(pattern, file_path):
            return True
    return False


def get_remote_files(sftp: paramiko.SFTPClient, remote_dir: str) -> dict:
    """
    Рекурсивно получает список всех файлов и директорий на сервере с их атрибутами.
    Возвращает словарь {относительный_путь: атрибуты}.
    """
    remote_files = {}
    try:
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = os.path.join(remote_dir, entry.filename)
            rel_path = os.path.relpath(remote_path, remote_dir)
            remote_files[rel_path] = entry

            if stat.S_ISDIR(entry.st_mode):
                sub_files = get_remote_files(sftp, remote_path)
                remote_files.update({os.path.join(rel_path, k): v for k, v in sub_files.items()})
    except Exception as e:
        print(f"Ошибка при получении файлов из {remote_dir}: {e}")
    return remote_files


def sync_files() -> None:
    """
    Основная функция синхронизации:
    - Подключается к серверу.
    - Синхронизирует .db файлы из REMOTE_DB_DIR в LOCAL_DB_DIR.
    - Синхронизирует .log файлы из REMOTE_LOG_DIR в LOCAL_LOG_DIR.
    - Удаляет локальные файлы/директории, которых нет на сервере или не соответствуют фильтрам.
    """
    # Подключение по SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    private_key = paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, pkey=private_key)

    # Создание SFTP-клиента
    sftp = ssh.open_sftp()

    try:
        # Синхронизация .db файлов
        print("Синхронизация .db файлов...")
        remote_db_files = get_remote_files(sftp, REMOTE_DB_DIR)
        print(f"Найдено .db файлов на сервере: {len(remote_db_files)}")

        # Получаем список локальных .db файлов
        local_db_files = set()
        for root, dirs, files in os.walk(LOCAL_DB_DIR):
            for file in files:
                local_path = Path(root) / file
                rel_path = str(local_path.relative_to(LOCAL_DB_DIR)).replace("\\", "/")
                local_db_files.add(rel_path)
            for dir in dirs:
                local_dir_path = Path(root) / dir
                rel_dir_path = str(local_dir_path.relative_to(LOCAL_DB_DIR)).replace("\\", "/")
                local_db_files.add(rel_dir_path + "/")

        # Синхронизация .db: скачивание и создание директорий
        for rel_path, attr in remote_db_files.items():
            local_path = LOCAL_DB_DIR / rel_path.replace("/", os.sep)
            if stat.S_ISDIR(attr.st_mode):
                local_path.mkdir(parents=True, exist_ok=True)
                print(f"Создана директория для .db: {local_path}")
            elif is_included(rel_path, DB_INCLUDE_PATTERNS):
                remote_path = os.path.join(REMOTE_DB_DIR, rel_path)
                if not local_path.exists() or os.path.getmtime(local_path) < attr.st_mtime:
                    sftp.get(remote_path, str(local_path))
                    print(f"Скачан .db файл: {local_path}")
                os.utime(local_path, (attr.st_atime, attr.st_mtime))

        # Удаление локальных .db файлов/директорий, которых нет на сервере
        for local_rel_path in list(local_db_files):
            if local_rel_path.endswith("/"):
                local_rel_path = local_rel_path[:-1]
            if local_rel_path not in remote_db_files and not any(local_rel_path.startswith(k + "/") for k in remote_db_files if remote_db_files[k].st_mode & stat.S_IFDIR):
                local_path = LOCAL_DB_DIR / local_rel_path.replace("/", os.sep)
                if local_path.is_file():
                    local_path.unlink()
                    print(f"Удалён .db файл: {local_path}")
                elif local_path.is_dir():
                    try:
                        local_path.rmdir()
                        print(f"Удалена директория для .db: {local_path}")
                    except OSError:
                        pass  # Пропускаем непустые директории

        # Синхронизация .log файлов
        print("Синхронизация .log файлов...")
        remote_log_files = get_remote_files(sftp, REMOTE_LOG_DIR)
        print(f"Найдено .log файлов на сервере: {len(remote_log_files)}")

        # Получаем список локальных .log файлов
        local_log_files = set()
        for root, dirs, files in os.walk(LOCAL_LOG_DIR):
            for file in files:
                local_path = Path(root) / file
                rel_path = str(local_path.relative_to(LOCAL_LOG_DIR)).replace("\\", "/")
                local_log_files.add(rel_path)
            for dir in dirs:
                local_dir_path = Path(root) / dir
                rel_dir_path = str(local_dir_path.relative_to(LOCAL_LOG_DIR)).replace("\\", "/")
                local_log_files.add(rel_dir_path + "/")

        # Синхронизация .log: скачивание и создание директорий
        for rel_path, attr in remote_log_files.items():
            local_path = LOCAL_LOG_DIR / rel_path.replace("/", os.sep)
            if stat.S_ISDIR(attr.st_mode):
                local_path.mkdir(parents=True, exist_ok=True)
                print(f"Создана директория для .log: {local_path}")
            elif is_included(rel_path, LOG_INCLUDE_PATTERNS):
                remote_path = os.path.join(REMOTE_LOG_DIR, rel_path)
                if not local_path.exists() or os.path.getmtime(local_path) < attr.st_mtime:
                    sftp.get(remote_path, str(local_path))
                    print(f"Скачан .log файл: {local_path}")
                os.utime(local_path, (attr.st_atime, attr.st_mtime))

        # Удаление локальных .log файлов/директорий, которых нет на сервере
        for local_rel_path in list(local_log_files):
            if local_rel_path.endswith("/"):
                local_rel_path = local_rel_path[:-1]
            if local_rel_path not in remote_log_files and not any(local_rel_path.startswith(k + "/") for k in remote_log_files if remote_log_files[k].st_mode & stat.S_IFDIR):
                local_path = LOCAL_LOG_DIR / local_rel_path.replace("/", os.sep)
                if local_path.is_file():
                    local_path.unlink()
                    print(f"Удалён .log файл: {local_path}")
                elif local_path.is_dir():
                    try:
                        local_path.rmdir()
                        print(f"Удалена директория для .log: {local_path}")
                    except OSError:
                        pass  # Пропускаем непустые директории

    except Exception as e:
        print(f"Ошибка синхронизации: {e}")
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    # Создаём локальные директории, если не существуют
    LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    sync_files()