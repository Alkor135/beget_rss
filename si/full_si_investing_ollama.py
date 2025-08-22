# coding: utf-8
import subprocess
from pathlib import Path
from datetime import datetime

# Получаем текущую дату и время для логов
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Путь к интерпретатору Python в виртуальном окружении
python_exe = Path(r"C:\Users\Alkor\PycharmProjects\beget_rss\.venv\Scripts\python.exe")

# Список Python-скриптов для запуска
python_scripts = [
    r"C:\Users\Alkor\PycharmProjects\beget_rss\si\si_download_minutes_to_db.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\si\si_21_00_convert_minutes_to_days.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\si\si_21_00_db_investing_month_to_md.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\si\predict_next_session_investing_ollama.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\si\backtesting_investing_ollama.py"
]

# Проверка существования интерпретатора Python
def check_python_exe():
    timestamp = get_timestamp()
    if not python_exe.exists():
        error_msg = f"[{timestamp}] Ошибка: Интерпретатор Python не найден в {python_exe}"
        print(error_msg)
        log_dir = Path(r"C:\Users\Alkor\gd\predict_ai\si_investing_ollama\log")
        ensure_dir(log_dir)
        with open(log_dir / "sync.log", 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")
        exit(1)

# Создание директории, если она не существует
def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

# Запуск Python-скриптов
def run_python_scripts():
    log_dir = Path(r"C:\Users\Alkor\gd\predict_ai\si_investing_ollama\log")
    ensure_dir(log_dir)
    sync_log = log_dir / "sync.log"

    for script in python_scripts:
        timestamp = get_timestamp()
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
    run_python_scripts()