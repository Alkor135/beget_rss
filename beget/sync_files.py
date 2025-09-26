"""
Скрипт для синхронизации файлов .db и .log с удалённого сервера на локальную машину.
Эквивалент PowerShell-скрипта sync_files.ps1.
Использует WSL и rsync для синхронизации данных.
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
        "name": "investing",
        "db_dir": r"C:\Users\Alkor\gd\db_rss_investing",
        "log_dir": r"C:\Users\Alkor\gd\db_rss_investing\log",
        "db_remote": "/home/user/rss_scraper/db_rss_investing/",
        "log_remote": "/home/user/rss_scraper/log/",
        "log_pattern": "rss_scraper_investing_month*.log"
    },
    {
        "name": "interfax",
        "db_dir": r"C:\Users\Alkor\gd\db_rss_interfax",
        "log_dir": r"C:\Users\Alkor\gd\db_rss_interfax\log",
        "db_remote": "/home/user/rss_scraper/db_rss_interfax/",
        "log_remote": "/home/user/rss_scraper/log/",
        "log_pattern": "rss_scraper_interfax_month*.log"
    },
    {
        "name": "prime",
        "db_dir": r"C:\Users\Alkor\gd\db_rss_prime",
        "log_dir": r"C:\Users\Alkor\gd\db_rss_prime\log",
        "db_remote": "/home/user/rss_scraper/db_rss_prime/",
        "log_remote": "/home/user/rss_scraper/log/",
        "log_pattern": "rss_scraper_prime_month*.log"
    }
]

# Создание директории, если она не существует
def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

# Выполнение команды rsync с логированием
def run_rsync(command, log_file, section_name, timestamp):
    try:
        timestamp = get_timestamp()
        print(f"[{timestamp}] Начало выполнения: {section_name}")
        print(f"[{timestamp}] Команда: {' '.join(command)}")  # Выводим полную команду

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
            f.write(f"[{timestamp}] {section_name}\n")
            for line in result.stdout.splitlines():
                f.write(f"[{timestamp}] {line}\n")
            if result.stderr:
                f.write(f"[{timestamp}] Ошибка:\n{result.stderr}\n")

        if result.returncode == 0:
            print(f"[{timestamp}] {section_name} успешно выполнен")
        else:
            print(f"[{timestamp}] {section_name} завершён с кодом {result.returncode}")

    except subprocess.CalledProcessError as e:
        error_msg = f"[{timestamp}] Ошибка при выполнении {section_name}: {e.stderr}"
        print(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")

    except subprocess.TimeoutExpired:
        error_msg = f"[{timestamp}] Таймаут команды {section_name}"
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
        timestamp = get_timestamp()
        print(f"[{get_timestamp()}] Запуск синхронизации .db файлов")
        with open(log_file, 'w', encoding='utf-8') as f:
            # f.write(f"[{timestamp}] Sync .db files\n")
            f.write(f"[{timestamp}] Синхронизация .db файлов\n")
        rsync_db_cmd = [
            "wsl", "rsync", "-avz",
            "--include=*/", "--include=**/*.db", "--exclude=*",
            f"root@109.172.46.10:{config['db_remote']}",
            f"/mnt/c{str(db_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_db_cmd, log_file, "Sync .db files", timestamp)

        # Синхронизация .log файлов
        timestamp = get_timestamp()
        print(f"[{timestamp}] Запуск синхронизации .log файлов")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] Синхронизация .log файлов\n")
        rsync_log_cmd = [
            "wsl", "rsync", "-avz",
            f"--include={config['log_pattern']}", "--exclude=*",
            f"root@109.172.46.10:{config['log_remote']}",
            f"/mnt/c{str(log_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_log_cmd, log_file, "Sync .log files", timestamp)

if __name__ == "__main__":
    sync_files()