"""
Мастер-скрипт для последовательного запуска всех рабочих скриптов.
Запускается Планировщиком задач через одно задание.
"""

import subprocess
import sys
import os

BASE = r"C:\Users\Alkor\PycharmProjects\beget_rss"
PYTHON = os.path.join(BASE, ".venv", "Scripts", "python.exe")

# список скриптов по порядку
SCRIPTS = [
    r"beget\sync_files.py",

    r"rts\rts_download_minutes_to_db.py",
    r"rts\rts_21_00_convert_minutes_to_days.py",
    r"rts\rts_21_00_db_investing_month_to_md.py",
    r"rts\rts_predict_next_session_investing_ollama.py",
    r"rts\rts_backtesting_investing_ollama.py",
    r"rts\rts_backtest_multi_max_investing.py",
    # r"trade\trade_rts_tri.py",

    r"mix\mix_download_minutes_to_db.py",
    r"mix\mix_21_00_convert_minutes_to_days.py",
    r"mix\mix_21_00_db_investing_month_to_md.py",
    r"mix\mix_predict_next_session_investing_ollama.py",
    r"mix\mix_backtesting_investing_ollama.py",
    r"mix\mix_backtest_multi_max_investing.py",
    # r"trade\trade_mix_tri.py",

    r"ng\ng_download_minutes_to_db.py",
    r"ng\ng_21_00_convert_minutes_to_days.py",
    r"ng\ng_21_00_db_investing_month_to_md.py",
    r"ng\ng_predict_next_session_investing_ollama.py",
    r"ng\ng_backtesting_investing_ollama.py",
    r"ng\ng_backtest_multi_max_investing.py",
    # r"trade\trade_ng_tri.py",

    r"br\br_download_minutes_to_db.py",
    r"br\br_21_00_convert_minutes_to_days.py",
    r"br\br_21_00_db_investing_month_to_md.py",
    r"br\br_predict_next_session_investing_ollama.py",
    r"br\br_backtesting_investing_ollama.py",
    r"br\br_backtest_multi_max_investing.py",

    r"gold\gold_download_minutes_to_db.py",
    r"gold\gold_21_00_convert_minutes_to_days.py",
    r"gold\gold_21_00_db_investing_month_to_md.py",
    r"gold\gold_predict_next_session_investing_ollama.py",
    r"gold\gold_backtesting_investing_ollama.py",
    r"gold\gold_backtest_multi_max_investing.py",

    r"si\si_download_minutes_to_db.py",
    r"si\si_21_00_convert_minutes_to_days.py",
    r"si\si_21_00_db_investing_month_to_md.py",
    r"si\si_predict_next_session_investing_ollama.py",
    r"si\si_backtesting_investing_ollama.py",
    r"si\si_backtest_multi_max_investing.py",

    r"spyf\spyf_download_minutes_to_db.py",
    r"spyf\spyf_21_00_convert_minutes_to_days.py",
    r"spyf\spyf_21_00_db_investing_month_to_md.py",
    r"spyf\spyf_predict_next_session_investing_ollama.py",
    r"spyf\spyf_backtesting_investing_ollama.py",
    r"spyf\spyf_backtest_multi_max_investing.py",

    r"rts/test_01/simulate_trade_01.py",

    r"mix/test_01/simulate_trade_01.py",

    r"ng/test_01/simulate_trade_01.py",
]

def run_script(script: str) -> int:
    script_path = os.path.join(BASE, script)
    cwd = os.path.dirname(script_path)
    print(f"\n=== Запуск: {script} ===")
    result = subprocess.run([PYTHON, script_path], cwd=cwd)
    return result.returncode

def main():
    for script in SCRIPTS:
        code = run_script(script)
        if code != 0:
            print(f"❌ Ошибка выполнения {script}, код {code}")
            os.system("pause")
            sys.exit(code)
    print("\n✅ Все скрипты выполнены успешно")
    input("\nНажмите Enter для выхода...")  # вместо sys.exit вручную
    os.system("pause")

if __name__ == "__main__":
    main()
