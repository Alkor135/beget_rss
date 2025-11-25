import pandas as pd
from pathlib import Path
import pickle, hashlib, numpy as np, sqlite3, logging, yaml
from datetime import datetime
from langchain_core.documents import Document

# === === === Настройки === === ===
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings.get('ticker', "RTS")  # Тикер инструмента
ticker_lc = ticker.lower()
provider = settings.get('provider', 'investing')  # Провайдер RSS новостей
url_ai = settings['url_ai']  # Ollama API без тайм-аута
model_name = settings['model_name']  # Ollama модель
min_prev_files = settings['min_prev_files']  # Минимальное количество предыдущих файлов
max_prev_files = settings['max_prev_files']  # Максимальное количество предыдущих файлов

md_path = Path(  # Путь к markdown-файлам
    settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(
    settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
result_file = Path(  # Итоговый XLSX файл
    fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\test'
    fr'\{ticker_lc}_backtest_results_{provider}_ollama.xlsx')  # Путь к итоговому XLSX файлу

def load_quotes(path_db_quote):
    """Читает таблицу Futures из базы данных котировок и возвращает DataFrame с next_bar_pips."""
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)
    df = df.sort_values('TRADEDATE', ascending=True)
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    # Вычисляем направление и количество пунктов бара за дату
    df['pips'] = df.apply(lambda x: (x['CLOSE'] - x['OPEN']), axis=1)
    # Вычисляем направление и количество пунктов за следующую дату
    df['next_bar_pips'] = df.apply(
        lambda x: (x['CLOSE'] - x['OPEN']), axis=1
    ).shift(-1)
    # Удаляем строки с NaN в next_bar_pips, если нужно
    df = df.dropna(subset=['next_bar_pips'])
    return df[['TRADEDATE', 'pips', 'next_bar_pips']].set_index('TRADEDATE')

def main():
    quotes = load_quotes(path_db_day)
    print(quotes)
    # docs = load_md_files(md_path)
    # cache = load_cache(cache_file)

if __name__ == '__main__':
    main()