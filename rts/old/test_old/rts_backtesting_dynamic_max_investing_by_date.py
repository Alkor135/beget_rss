#!/usr/bin/env python3
"""
rts_backtesting_dynamic_max_investing_by_date.py
================================================
Бэктест с переменным количеством предыдущих файлов (max_prev_files),
определяемым для каждой даты индивидуально на основе результатов из
'rts_backtest_multi_max_investing_by_date.py'.
"""

import pandas as pd
from pathlib import Path
import pickle, hashlib, numpy as np, sqlite3, logging, yaml
from datetime import datetime
from langchain_core.documents import Document

# === Загрузка настроек ===
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings.get('ticker', "RTS")
ticker_lc = ticker.lower()
provider = settings.get('provider', 'investing')

md_path = Path(settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
output_dir = Path(settings['output_dir'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))

# === Вводим дату начала симуляции ===
start_date = settings.get('start_date', '2025-08-01')

# === Файл с результатами мульти-теста ===
multi_max_file = Path(
    fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\{ticker_lc}_backtest_results_multi_max_{provider}_*.xlsx'
)

# === Настройка логирования ===
log_file = output_dir / 'log' / f'{ticker_lc}_backtesting_dynamic_{provider}.txt'
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(logging.StreamHandler())
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
logger.addHandler(file_handler)

# === Косинусное сходство ===
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

# === Загрузка котировок ===
def load_quotes(path_db):
    with sqlite3.connect(path_db) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['next_bar_pips'] = (df['CLOSE'] - df['OPEN']).shift(-1)
    return df.dropna(subset=['next_bar_pips']).set_index('TRADEDATE')

# === Загрузка markdown-файлов ===
def load_md_files(directory):
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
    docs = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        if not content.startswith('---'): continue
        parts = content.split('---', 2)
        if len(parts) < 3: continue
        metadata = yaml.safe_load(parts[1]) or {}
        docs.append(Document(
            page_content=parts[2].strip(),
            metadata={
                "next_bar": str(metadata.get("next_bar", "unknown")),
                "date": f.stem,
                "md5": hashlib.md5(content.encode('utf-8')).hexdigest()
            }
        ))
    return docs

# === Загрузка кэша эмбеддингов ===
def load_cache(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# === Загрузка max_prev_value по датам из Excel ===
def load_dynamic_max_values(path_pattern):
    files = list(Path(path_pattern.parent).glob(path_pattern.name))
    if not files:
        logger.error(f"Файл с multi_max результатами не найден по шаблону: {path_pattern}")
        exit(1)
    df = pd.read_excel(files[0])
    last_row = df.sort_values('test_date').iloc[-1]
    max_cols = [c for c in df.columns if c.startswith('max_')]
    per_date = {}
    for _, row in df.iterrows():
        max_col = row[max_cols].idxmax()
        per_date[row['test_date']] = int(max_col.split('_')[1])
    return per_date

# === Основной процесс ===
def main():
    quotes = load_quotes(path_db_day)
    docs = load_md_files(md_path)
    cache = load_cache(cache_file)
    dynamic_max = load_dynamic_max_values(multi_max_file)

    results = []
    for doc in docs:
        test_date = doc.metadata['date']
        if test_date < start_date:
            continue
        max_prev = dynamic_max.get(test_date)
        if not max_prev:
            continue

        test_embedding = next((i['embedding'] for i in cache if i['id'] == doc.metadata['md5']), None)
        if test_embedding is None:
            continue

        prev_cache = sorted(
            [i for i in cache if i['metadata']['date'] < test_date],
            key=lambda x: x['metadata']['date'], reverse=True
        )[:max_prev]

        if not prev_cache:
            continue

        sims = [(cosine_similarity(test_embedding, i['embedding']), i['metadata']) for i in prev_cache]
        sims.sort(key=lambda x: x[0], reverse=True)
        predicted = sims[0][1]['next_bar']
        real = doc.metadata['next_bar']
        is_correct = predicted == real
        try:
            pips = abs(quotes.loc[test_date, 'next_bar_pips'])
            pips = pips if is_correct else -pips
        except KeyError:
            continue

        results.append({
            'test_date': test_date,
            'predicted_next_bar': predicted,
            'real_next_bar': real,
            'is_correct': is_correct,
            'max_prev_files': max_prev,
            'next_bar_pips': pips
        })
        logger.info(f"{test_date}: max_prev={max_prev}, pred={predicted}, real={real}, ok={is_correct}")

    if results:
        df = pd.DataFrame(results)
        df['cumulative_next_bar_pips'] = df['next_bar_pips'].cumsum()
        out_file = output_dir / f'{ticker_lc}_backtest_results_dynamic_{provider}.xlsx'
        df.to_excel(out_file, index=False, engine='openpyxl')
        logger.info(f"✅ Результаты сохранены в {out_file}")

if __name__ == '__main__':
    main()
