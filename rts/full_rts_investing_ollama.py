# Скрипт для последовательного запуска зависимых Python-скриптов.
# Передаёт параметры num_mds=20 и num_dbs=2 для rts_21_00_db_investing_month_to_md.
# Логирует выполнение и ошибки в C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log.

from rts_download_minutes_to_db import main as download_minutes
from rts_21_00_convert_minutes_to_days import main as convert_to_days
from rts_21_00_db_investing_month_to_md import main as convert_to_md
from rts_predict_next_session_investing_ollama import main as predict_session
from rts_backtesting_investing_ollama import main as backtest
from rts_backtest_multi_max_investing import main as backtest_multi
from pathlib import Path
from datetime import datetime
import shutil
import os

# Получаем текущую дату и время для логов
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def remove_pycache(directory):
    # Проходим по всем файлам и папкам в указанной директории
    for root, dirs, files in os.walk(directory, topdown=False):
        # Проверяем, есть ли папка __pycache__ в списке директорий
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                # Удаляем папку __pycache__ и её содержимое
                shutil.rmtree(pycache_path)
                print(f"Удалена папка: {pycache_path}")
            except Exception as e:
                print(f"Ошибка при удалении {pycache_path}: {e}")

# Создание директории, если она не существует
def ensure_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

def run_python_scripts():
    """Для последовательного запуска скриптов"""
    root_directory = os.getcwd()  # Указываем текущую директорию для удаления __pycache__
    remove_pycache(root_directory)  # Удаляем __pycache__

    log_dir = Path(r"C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log")
    ensure_dir(log_dir)
    full_log = log_dir / "full.log"

    for func, name, kwargs in [
        (download_minutes, "rts_download_minutes_to_db", {}),
        (convert_to_days, "rts_21_00_convert_minutes_to_days", {}),
        (convert_to_md, "rts_21_00_db_investing_month_to_md", {"num_mds": 20, "num_dbs": 2}),
        (predict_session, "rts_predict_next_session_investing_ollama", {"max_prev_files": 8}),
        (backtest, "rts_backtesting_investing_ollama", {"max_prev_files": 8}),
        (backtest_multi, "rts_backtest_multi_max_investing", {})
    ]:
        timestamp = get_timestamp()
        # script_log = log_dir / f"{name}.log"
        with open(full_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Запуск {name}.py\n")
        try:
            # with open(script_log, 'w', encoding='utf-8') as f:
                print(f"[{timestamp}] Запуск {name}.py")
                func(**kwargs)  # Вызов функции с параметрами
                timestamp = get_timestamp()
                success_msg = f"[{timestamp}] {name}.py успешно выполнен"
                print(success_msg)
                # f.write(f"[{timestamp}] {success_msg}\n")
                with open(full_log, 'a', encoding='utf-8') as f:
                    f.write(success_msg + "\n")
        except Exception as e:
            error_msg = f"[{timestamp}] Ошибка при выполнении {name}.py: {str(e)}"
            print(error_msg)
            with open(full_log, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
            # with open(script_log, 'w', encoding='utf-8') as f:
            #     f.write(f"[{timestamp}] [{name}.py] ERROR: {str(e)}\n")
    with open(full_log, 'a', encoding='utf-8') as f:
        f.write(f"\n")

if __name__ == "__main__":
    run_python_scripts()