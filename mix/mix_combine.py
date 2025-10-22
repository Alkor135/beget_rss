#!/usr/bin/env python3
"""
rts_backtesting_dynamic_max_investing_by_date.py
================================================
Комбинированный бэктест:
1. Рассчитывает cumulative_next_bar_pips для max_prev_files ∈ [3..30].
2. Определяет оптимальный max_prev_files (max_prev_value) для каждой даты.
3. Запускает симуляцию торговли с динамическим max_prev_files.
"""

import pandas as pd
from pathlib import Path
import pickle, hashlib, numpy as np, sqlite3, logging, yaml
from datetime import datetime
from langchain_core.documents import Document

# === === === Настройки === === ===
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings.get('ticker', "MIX")
ticker_lc = ticker.lower()
provider = settings.get('provider', 'investing')
min_prev_files = settings.get('min_prev_files', 2)
start_date = settings.get('start_date', '2025-07-01')

md_path = Path(settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
output_dir = Path(settings['output_dir'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))

# === Логирование ===
log_file = output_dir / 'log' / f'{ticker_lc}_backtesting_dynamic_{provider}.txt'
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
console = logging.StreamHandler()
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(fmt)
file_handler.setFormatter(fmt)
logger.addHandler(console)
logger.addHandler(file_handler)

# === Вспомогательные функции ===
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

def load_quotes(path_db):
    with sqlite3.connect(path_db) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)
    df['TRADEDATE'] = df['TRADEDATE'].astype(str)
    df['next_bar_pips'] = (df['CLOSE'] - df['OPEN']).shift(-1)
    return df.dropna(subset=['next_bar_pips']).set_index('TRADEDATE')

def load_md_files(directory):
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
    docs = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        if not content.startswith('---'):
            continue
        parts = content.split('---', 2)
        if len(parts) < 3:
            continue
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

def load_cache(path):
    if not path.exists():
        logger.error(f"Кэш эмбеддингов не найден: {path}")
        exit(1)
    with open(path, 'rb') as f:
        return pickle.load(f)

# === Расчёт cumulative_next_bar_pips для всех max_prev_files ===
def compute_multi_max(documents, cache, quotes_df, min_prev_files):
    logger.info("=== Этап 1: расчёт multi-max для диапазона 3..30 ===")
    all_results = pd.DataFrame()
    for max_prev in range(3, 31):
        results = []
        for test_idx in range(min_prev_files, len(documents)):
            test_doc = documents[test_idx]
            test_date = test_doc.metadata['date']
            real = test_doc.metadata['next_bar']
            if real in ['unknown', 'None']:
                continue

            test_embedding = next((i['embedding'] for i in cache if i['id'] == test_doc.metadata['md5']), None)
            if test_embedding is None:
                continue

            prev_cache = sorted(
                [i for i in cache if i['metadata']['date'] < test_date],
                key=lambda x: x['metadata']['date'], reverse=True
            )[:max_prev]
            if len(prev_cache) < min_prev_files:
                continue

            sims = [(cosine_similarity(test_embedding, i['embedding']), i['metadata']) for i in prev_cache]
            sims.sort(key=lambda x: x[0], reverse=True)
            pred = sims[0][1]['next_bar']
            is_correct = pred == real
            try:
                pips = abs(quotes_df.loc[test_date, 'next_bar_pips'])
                pips = pips if is_correct else -pips
            except KeyError:
                continue

            results.append({'test_date': test_date, 'next_bar_pips': pips})

        df = pd.DataFrame(results)
        if not df.empty:
            df['cumulative_next_bar_pips'] = df['next_bar_pips'].cumsum()
            df = df.rename(columns={'cumulative_next_bar_pips': f'max_{max_prev}'})
            df = df[['test_date', f'max_{max_prev}']]
            if all_results.empty:
                all_results = df
            else:
                all_results = all_results.merge(df, on='test_date', how='outer')
        logger.info(f"→ рассчитано для max_prev={max_prev}")
    return all_results

# === Основной процесс ===
def main():
    quotes = load_quotes(path_db_day)
    docs = load_md_files(md_path)
    cache = load_cache(cache_file)

    # Этап 1: рассчитываем cumulative для каждого max_prev_files
    all_results = compute_multi_max(docs, cache, quotes, min_prev_files)

    if all_results.empty:
        logger.error("Нет данных для расчёта multi-max.")
        return

    # Этап 2: определяем max_prev_value по каждой дате
    logger.info("=== Этап 2: вычисление max_prev_value для каждой даты ===")
    max_cols = [c for c in all_results.columns if c.startswith('max_')]
    per_date_max = {}
    for _, row in all_results.iterrows():
        valid = row[max_cols].dropna()
        if valid.empty:
            continue
        col = valid.idxmax()
        per_date_max[row['test_date']] = int(col.split('_')[1])

    # Этап 3: динамическая симуляция
    logger.info("=== Этап 3: симуляция торговли с динамическим max_prev_files ===")
    results = []
    for doc in docs:
        test_date = doc.metadata['date']
        if test_date < start_date or test_date not in per_date_max:
            continue
        max_prev = per_date_max[test_date]
        real = doc.metadata['next_bar']
        if real in ['unknown', 'None']:
            continue

        test_embedding = next((i['embedding'] for i in cache if i['id'] == doc.metadata['md5']), None)
        if test_embedding is None:
            continue

        prev_cache = sorted(
            [i for i in cache if i['metadata']['date'] < test_date],
            key=lambda x: x['metadata']['date'], reverse=True
        )[:max_prev]
        if len(prev_cache) < min_prev_files:
            continue

        sims = [(cosine_similarity(test_embedding, i['embedding']), i['metadata']) for i in prev_cache]
        sims.sort(key=lambda x: x[0], reverse=True)
        pred = sims[0][1]['next_bar']
        is_correct = pred == real

        try:
            pips = abs(quotes.loc[test_date, 'next_bar_pips'])
            pips = pips if is_correct else -pips
        except KeyError:
            continue

        results.append({
            'test_date': test_date,
            'predicted_next_bar': pred,
            'real_next_bar': real,
            'is_correct': is_correct,
            'max_prev_files': max_prev,
            'next_bar_pips': pips
        })
        logger.info(f"{test_date}: max_prev={max_prev}, pred={pred}, real={real}, ok={is_correct}")

    if not results:
        logger.error("Нет результатов для симуляции.")
        return

    df = pd.DataFrame(results)
    df['cumulative_next_bar_pips'] = df['next_bar_pips'].cumsum()
    # out_file = output_dir / f'{ticker_lc}_backtest_results_dynamic_{provider}.xlsx'
    out_file = f'{ticker_lc}_backtest_results_dynamic_{provider}.xlsx'
    df.to_excel(out_file, index=False, engine='openpyxl')
    logger.info(f"✅ Результаты сохранены в {out_file}")

if __name__ == '__main__':
    main()
