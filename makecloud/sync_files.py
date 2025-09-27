"""
Скрипт для синхронизации файлов .db и .log с удалённого сервера на локальную машину.
Использует WSL и rsync для синхронизации данных.
Нужно прописать в WSL figerprint для сервера. Запустить вручную на WSL:
rsync -avz --include=*/ --include=**/*.db --exclude=* ubuntu@212.22.94.68:/home/ubuntu/rss_scraper/db_data /mnt/c/Users/Alkor/gd/db_rss/
"""

import subprocess
from pathlib import Path
from datetime import datetime

# Получаем текущую дату и время для логов
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Конфигурация синхронизации
sync_configs = [
    {
        "name": "rss_news",
        "db_dir": r"C:\Users\Alkor\gd\db_rss",
        "log_dir": r"C:\Users\Alkor\gd\db_rss\log",
        "db_remote": "/home/ubuntu/rss_scraper/db_data/",
        "log_remote": "/home/ubuntu/rss_scraper/log/",
        "log_pattern": "*.txt"
    }
]

# Создание директории, если она не существует
def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

# Выполнение команды rsync с логированием и таймаутом
def run_rsync(command, log_file, section_name):  # , timestamp
    try:
        print(f"[{get_timestamp()}] Начало выполнения: {section_name}")
        print(f"[{get_timestamp()}] Команда: {' '.join(command)}")  # Выводим полную команду

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=False,  # Не прерываем на ошибке для получения вывода
            timeout=30    # Добавлен таймаут 30 секунд
        )

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{get_timestamp()}] {section_name}\n")
            for line in result.stdout.splitlines():
                f.write(f"[{get_timestamp()}] {line}\n")
            if result.stderr:
                f.write(f"[{get_timestamp()}] Ошибка:\n{result.stderr}\n")

        if result.returncode == 0:
            print(f"[{get_timestamp()}] {section_name} успешно выполнен")
        else:
            print(f"[{get_timestamp()}] {section_name} завершён с кодом {result.returncode}")

    except subprocess.CalledProcessError as e:
        error_msg = f"[{get_timestamp()}] Ошибка при выполнении {section_name}: {e.stderr}"
        print(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")

    except subprocess.TimeoutExpired:
        error_msg = f"[{get_timestamp()}] Таймаут команды {section_name}"
        print(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")

# Синхронизация файлов
def sync_files():
    for config in sync_configs:
        log_dir = Path(config["log_dir"])
        log_file = log_dir / "sync.log"
        db_dir = Path(config["db_dir"])

        # Создаём директорию для логов
        ensure_dir(log_dir)

        # Синхронизация .db файлов
        # timestamp = get_timestamp()
        print(f"[{get_timestamp()}] Запуск синхронизации .db файлов")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"[{get_timestamp()}] Синхронизация .db файлов\n")
        rsync_db_cmd = [
            "wsl", "rsync", "-avz", "--progress",
            "--include=*/", "--include=**/*.db", "--exclude=*",
            f"ubuntu@212.22.94.68:{config['db_remote']}",
            f"/mnt/c{str(db_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_db_cmd, log_file, "Sync .db files")

        # Синхронизация .log файлов
        # timestamp = get_timestamp()
        print(f"[{get_timestamp()}] Запуск синхронизации .log файлов")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{get_timestamp()}] Синхронизация .log файлов\n")
        rsync_log_cmd = [
            "wsl", "rsync", "-avz", "--progress",
            "--include=*/",  # Включаем все подпапки
            f"--include={config['log_pattern']}",  # Включаем нужные .txt файлы
            "--exclude=*",  # Исключаем всё остальное
            f"ubuntu@212.22.94.68:{config['log_remote']}",
            f"/mnt/c{str(log_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_log_cmd, log_file, "Sync .log files")

if __name__ == "__main__":
    sync_files()