"""
Скрипт для синхронизации файлов .db и .log с удалённого сервера на локальную машину.
Эквивалент PowerShell-скрипта sync_files.ps1.
Использует WSL и rsync для синхронизации данных.
"""

import subprocess
from pathlib import Path
from datetime import datetime
import sys

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
        "log_pattern": "rss_scraper_investing_to_db_month_msk.txt"
    },
    {
        "name": "interfax",
        "db_dir": r"C:\Users\Alkor\gd\db_rss_interfax",
        "log_dir": r"C:\Users\Alkor\gd\db_rss_interfax\log",
        "db_remote": "/home/user/rss_scraper/db_rss_interfax/",
        "log_remote": "/home/user/rss_scraper/log/",
        "log_pattern": "rss_scraper_interfax_to_db_month_msk.txt"
    },
    {
        "name": "prime",
        "db_dir": r"C:\Users\Alkor\gd\db_rss_prime",
        "log_dir": r"C:\Users\Alkor\gd\db_rss_prime\log",
        "db_remote": "/home/user/rss_scraper/db_rss_prime/",
        "log_remote": "/home/user/rss_scraper/log/",
        "log_pattern": "rss_scraper_prime_to_db_month_msk.txt"
    }
]

# Создание директории, если она не существует
def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

# Выполнение команды rsync с логированием
def run_rsync(command, log_file: Path, section_name: str):
    print(f"[{get_timestamp()}] Запуск: {section_name}")
    print(f"[{get_timestamp()}] Команда: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=600   # 10 минут
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{get_timestamp()}] --- {section_name} ---\n")
            if result.stdout:
                for line in result.stdout.splitlines():
                    f.write(f"[{get_timestamp()}] {line}\n")
            if result.stderr:
                f.write(f"[{get_timestamp()}] STDERR:\n{result.stderr}\n")

        if result.returncode == 0:
            print(f"[{get_timestamp()}] {section_name} успешно выполнен")
        else:
            print(f"[{get_timestamp()}] {section_name} завершён с кодом {result.returncode}")
            sys.exit(result.returncode)

    except subprocess.TimeoutExpired:
        error_msg = f"[{get_timestamp()}] Таймаут команды: {section_name}"
        print(error_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(error_msg + "\n")
        sys.exit(124)  # стандартный код таймаута

# Синхронизация файлов
def sync_files():
    for config in sync_configs:
        log_dir = Path(config["log_dir"])
        db_dir = Path(config["db_dir"])
        log_file = log_dir / "sync.log"

        ensure_dir(log_dir)

        # Синхронизация .db файлов
        print(f"[{get_timestamp()}] Синхронизация .db файлов ({config['name']})")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"[{get_timestamp()}] Синхронизация .db файлов\n")
        rsync_db_cmd = [
            "wsl", "rsync", "-avz", "--progress",
            "--include=*/", "--include=**/*.db", "--exclude=*",
            f"root@109.172.46.10:{config['db_remote']}",
            f"/mnt/c{str(db_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_db_cmd, log_file, f"Sync .db files: {config['name']}")

        # Синхронизация .log файлов
        print(f"[{get_timestamp()}] Синхронизация .log файлов ({config['name']})")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{get_timestamp()}] Синхронизация .log файлов\n")
        rsync_log_cmd = [
            "wsl", "rsync", "-avz", "--progress",
            f"--include={config['log_pattern']}", "--exclude=*",
            f"root@109.172.46.10:{config['log_remote']}",
            f"/mnt/c{str(log_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_log_cmd, log_file, f"Sync .log files: {config['name']}")

if __name__ == "__main__":
    sync_files()
