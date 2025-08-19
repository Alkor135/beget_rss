"""
Скрипт для синхронизации файлов .db и .log с удалённого сервера на локальную машину
и запуска Python-скриптов из виртуального окружения.
Эквивалент PowerShell-скрипта sync_files_01.ps1.
Исправлена обработка кодировки вывода для предотвращения UnicodeDecodeError.
"""

import subprocess
from pathlib import Path
from datetime import datetime
import os

# Получаем текущую дату и время для логов
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Путь к интерпретатору Python в виртуальном окружении
python_exe = Path(r"C:\Users\Alkor\PycharmProjects\beget_rss\.venv\Scripts\python.exe")

# Список Python-скриптов для запуска
python_scripts = [
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\rts_download_minutes_to_db.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\rts_21_00_convert_minutes_to_days.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\rts_21_00_db_investing_month_to_md.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\predict_next_session_investing_ollama.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\backtesting_investing_ollama.py"
]

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

# Проверка существования интерпретатора Python
def check_python_exe():
    timestamp = get_timestamp()
    if not python_exe.exists():
        error_msg = f"[{timestamp}] Ошибка: Интерпретатор Python не найден в {python_exe}"
        print(error_msg)
        with open(r"C:\Users\Alkor\gd\db_rss_investing\log\sync.log", 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")
        exit(1)

# Создание директории, если она не существует
def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

# Выполнение команды rsync с логированием
def run_rsync(command, log_file, section_name, timestamp):
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {section_name}\n")
            for line in result.stdout.splitlines():
                f.write(f"[{timestamp}] {line}\n")
        print(f"[{timestamp}] {section_name} успешно выполнен")
    except subprocess.CalledProcessError as e:
        error_msg = f"[{timestamp}] Ошибка при выполнении {section_name}: {e.stderr}"
        print(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")

# Синхронизация файлов
def sync_files():
    timestamp = get_timestamp()
    for config in sync_configs:
        log_dir = Path(config["log_dir"])
        log_file = log_dir / "sync.log"
        db_dir = Path(config["db_dir"])

        # Создаём директорию для логов
        ensure_dir(log_dir)

        # Синхронизация .db файлов
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Sync .db files ({config['name']})\n")
        rsync_db_cmd = [
            "wsl", "rsync", "-avz",
            "--include=*/", "--include=**/*.db", "--exclude=*",
            f"root@109.172.46.10:{config['db_remote']}",
            f"/mnt/c{str(db_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_db_cmd, log_file, f"Sync .db files ({config['name']})", timestamp)

        # Синхронизация .log файлов
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Sync .log files ({config['name']})\n")
        rsync_log_cmd = [
            "wsl", "rsync", "-avz",
            f"--include={config['log_pattern']}", "--exclude=*",
            f"root@109.172.46.10:{config['log_remote']}",
            f"/mnt/c{str(log_dir)[2:].replace('\\', '/')}/"
        ]
        run_rsync(rsync_log_cmd, log_file, f"Sync .log files ({config['name']})", timestamp)

# Запуск Python-скриптов
def run_python_scripts():
    timestamp = get_timestamp()
    log_dir = Path(r"C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log")
    ensure_dir(log_dir)
    sync_log = log_dir / "sync.log"

    for script in python_scripts:
        script_path = Path(script)
        if not script_path.exists():
            error_msg = f"[{timestamp}] Ошибка: Скрипт {script} не найден"
            print(error_msg)
            with open(sync_log, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
            continue

        script_name = script_path.name
        script_log = log_dir / f"{script_name.replace('.py', '.log')}"
        with open(sync_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Запуск {script_name}\n")

        try:
            with open(script_log, 'w', encoding='utf-8') as f:
                process = subprocess.run(
                    [str(python_exe), str(script_path)],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # Заменяем неподдерживаемые символы
                    check=True
                )
                if process.stdout:
                    for line in process.stdout.splitlines():
                        f.write(f"[{timestamp}] [{script_name}] {line}\n")
                if process.stderr:
                    for line in process.stderr.splitlines():
                        f.write(f"[{timestamp}] [{script_name}] ERROR: {line}\n")
            success_msg = f"[{timestamp}] {script_name} успешно выполнен"
            print(success_msg)
            with open(sync_log, 'a', encoding='utf-8') as f:
                f.write(success_msg + "\n")
        except subprocess.CalledProcessError as e:
            error_msg = f"[{timestamp}] Ошибка при выполнении {script_name}: {e.stderr if e.stderr else 'Неизвестная ошибка'}"
            print(error_msg)
            with open(sync_log, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
            with open(script_log, 'w', encoding='utf-8') as f:
                f.write(f"[{timestamp}] [{script_name}] ERROR: {e.stderr if e.stderr else 'Неизвестная ошибка'}\n")

if __name__ == "__main__":
    check_python_exe()
    sync_files()
    run_python_scripts()