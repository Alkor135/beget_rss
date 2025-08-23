"""
Скрипт для параллельного запуска Python-скриптов преобразования новостей в markdown.
Запускает скрипты группами по 3 с использованием ProcessPoolExecutor.
Логирует выполнение и ошибки в C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log.
"""

import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Получаем текущую дату и время для логов
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Путь к интерпретатору Python в виртуальном окружении
python_exe = Path(r"C:\Users\Alkor\PycharmProjects\beget_rss\.venv\Scripts\python.exe")

# Список Python-скриптов для параллельного запуска
python_scripts = [
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\rts_21_00_db_investing_month_to_md.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\mix_21_00_db_investing_month_to_md.py",
    r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\si_21_00_db_investing_month_to_md.py",
    # Добавьте другие скрипты сюда, например:
    # r"C:\Users\Alkor\PycharmProjects\beget_rss\rts\another_21_00_db_investing_month_to_md.py",
]

# Параметры для rts_21_00_db_investing_month_to_md.py и других (если требуется)
script_params = {
    "rts_21_00_db_investing_month_to_md.py": {"num_mds": 20, "num_dbs": 2},
    "mix_21_00_db_investing_month_to_md.py": {"num_mds": 20, "num_dbs": 2},
    "si_21_00_db_investing_month_to_md.py": {"num_mds": 20, "num_dbs": 2},
    # Добавьте параметры для других скриптов, если они отличаются
}

# Проверка существования интерпретатора Python
def check_python_exe():
    timestamp = get_timestamp()
    if not python_exe.exists():
        error_msg = f"[{timestamp}] Ошибка: Интерпретатор Python не найден в {python_exe}"
        print(error_msg)
        log_dir = Path(r"C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log")
        ensure_dir(log_dir)
        with open(log_dir / "sync.log", 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")
        exit(1)

# Создание директории, если она не существует
def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

# Запуск одного скрипта с параметрами
def run_script(script_path):
    timestamp = get_timestamp()
    script_path = Path(script_path)
    script_name = script_path.name
    log_dir = Path(r"C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log")
    ensure_dir(log_dir)
    sync_log = log_dir / "sync.log"
    script_log = log_dir / f"{script_name.replace('.py', '.log')}"

    if not script_path.exists():
        error_msg = f"[{timestamp}] Ошибка: Скрипт {script_path} не найден"
        print(error_msg)
        with open(sync_log, 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")
        return error_msg

    # Получение параметров для скрипта или пустой словарь
    params = script_params.get(script_name, {})
    arg_list = [str(python_exe), str(script_path)]
    for key, value in params.items():
        arg_list.extend([f"--{key}", str(value)])

    try:
        with open(script_log, 'w', encoding='utf-8') as f:
            process = subprocess.run(
                arg_list,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True
            )
            log_lines = [f"[{timestamp}] Запуск {script_name} с параметрами: {params}"]
            if process.stdout:
                log_lines.extend(f"[{timestamp}] [{script_name}] {line}" for line in process.stdout.splitlines())
            if process.stderr:
                log_lines.extend(f"[{timestamp}] [{script_name}] ERROR: {line}" for line in process.stderr.splitlines())
            success_msg = f"[{timestamp}] {script_name} успешно выполнен"
            log_lines.append(success_msg)
            print(success_msg)
            f.write("\n".join(log_lines) + "\n")
            with open(sync_log, 'a', encoding='utf-8') as f:
                f.write("\n".join(log_lines) + "\n")
        return success_msg
    except subprocess.CalledProcessError as e:
        error_msg = f"[{timestamp}] Ошибка при выполнении {script_name}: {e.stderr if e.stderr else 'Неизвестная ошибка'}"
        print(error_msg)
        log_lines = [f"[{timestamp}] Запуск {script_name} с параметрами: {params}", error_msg]
        with open(sync_log, 'a', encoding='utf-8') as f:
            f.write("\n".join(log_lines) + "\n")
        with open(script_log, 'w', encoding='utf-8') as f:
            f.write("\n".join(log_lines) + "\n")
        return error_msg

def run_parallel_scripts(max_workers=3):
    check_python_exe()
    log_dir = Path(r"C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log")
    ensure_dir(log_dir)
    sync_log = log_dir / "sync.log"
    timestamp = get_timestamp()
    with open(sync_log, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] Начало параллельного запуска скриптов\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_script = {executor.submit(run_script, script): script for script in python_scripts}
        for future in as_completed(future_to_script):
            script = future_to_script[future]
            try:
                result = future.result()
                print(f"[{timestamp}] Завершён: {script}")
            except Exception as e:
                error_msg = f"[{timestamp}] Ошибка при выполнении {script}: {str(e)}"
                print(error_msg)
                with open(sync_log, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")

    timestamp = get_timestamp()
    with open(sync_log, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] Параллельный запуск скриптов завершён\n")

if __name__ == "__main__":
    run_parallel_scripts(max_workers=3)