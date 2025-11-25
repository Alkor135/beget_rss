"""
Скрипт проводит мультибэктест прогнозов на основе косинусного сходства эмбеддингов markdown-отчётов.
Для каждой даты формирует .xlsx-файл с кумулятивным профитом (в пипсах) при разных значениях
max_prev_files (от 3 до 30). Использует кэш эмбеддингов, исторические котировки из SQLite и
скользящее окно последних test_days дней.
🔹 Вывод: один Excel-файл на дату — для анализа оптимальных параметров модели.
🔹 Назначение: сравнение эффективности прогнозов в зависимости от глубины истории.
"""

import pandas as pd
from pathlib import Path
import pickle
import hashlib
import numpy as np
import sqlite3
from langchain_core.documents import Document
import logging
import yaml


SETTINGS_FILE = Path(__file__).parent.parent / "settings.yaml"

with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings.get('ticker', "RTS")
ticker_lc = ticker.lower()
provider = settings.get('provider', 'investing')
min_prev_files = settings.get('min_prev_files', 2)
test_days = settings.get('test_days', 22) + 1   #  ← Добавлено

md_path = Path(settings['md_path'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
cache_file = Path(settings['cache_file'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
# output_dir = Path(settings['output_dir'].replace('{ticker_lc}', ticker_lc).replace('{provider}', provider))
xlsx_path = Path(__file__).parent / "xlsx_files"

log_file = Path(fr'{ticker_lc}_backtest_multi_max_{provider}_MULTI.txt')
log_file.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(logging.StreamHandler())
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
logger.addHandler(file_handler)


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return (dot_product / denom) if denom != 0 else 0.0


def load_markdown_files(directory):
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)
    documents = []
    for file_path in files:
        content = file_path.read_text(encoding='utf-8')

        md5_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1]
                text_content = parts[2]
                metadata = yaml.safe_load(metadata_yaml) or {}
                metadata_str = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),
                    "date": file_path.stem,
                    "source": file_path.name,
                    "md5": md5_hash
                }
                documents.append(Document(page_content=text_content, metadata=metadata_str))
    return documents


def load_quotes(path_db_quote):
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT TRADEDATE, OPEN, CLOSE FROM Futures", conn)

    df = df.sort_values('TRADEDATE')
    df['next_bar_pips'] = (df['CLOSE'] - df['OPEN']).shift(-1)
    df = df.dropna(subset=['next_bar_pips'])
    return df.set_index('TRADEDATE')[['next_bar_pips']]


def load_cache(cache_file):
    if not cache_file.exists():
        logger.error(f"❌ Кэш {cache_file} не найден.")
        exit(1)

    logger.info(f"Загрузка кэша {cache_file}")
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def backtest_for_docs(documents, cache, quotes_df, max_prev_files):
    results = []

    for test_doc in documents[min_prev_files:]:
        real_next_bar = test_doc.metadata['next_bar']
        test_date = test_doc.metadata['date']
        test_id = test_doc.metadata['md5']

        if real_next_bar in ('unknown', 'None'):
            continue

        test_emb = next((x['embedding'] for x in cache if x['id'] == test_id), None)
        if test_emb is None:
            continue

        prev_cache = sorted(
            [x for x in cache if x['metadata']['date'] < test_date],
            key=lambda x: x['metadata']['date'],
            reverse=True
        )[:max_prev_files]

        if len(prev_cache) < min_prev_files:
            continue

        sims = [
            (cosine_similarity(test_emb, x['embedding']), x['metadata'])
            for x in prev_cache
        ]
        sims.sort(key=lambda x: x[0], reverse=True)

        closest_metadata = sims[0][1]
        predicted_next_bar = closest_metadata['next_bar']
        is_correct = predicted_next_bar == real_next_bar

        try:
            p = quotes_df.loc[test_date, 'next_bar_pips']
        except KeyError:
            continue

        next_bar_pips = abs(p) if is_correct else -abs(p)

        results.append({
            'test_date': test_date,
            'value': next_bar_pips
        })

    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()

    df['cumulative'] = df['value'].cumsum()
    return df[['test_date', 'cumulative']]


def main():
    quotes_df = load_quotes(path_db_day)
    documents = load_markdown_files(md_path)
    cache = load_cache(cache_file)

    if len(documents) < 5:
        logger.error("Недостаточно markdown файлов (<5)")
        exit(1)

    for end_idx in range(4, len(documents)):
        cut_docs = documents[:end_idx + 1]
        date_str = documents[end_idx].metadata['date']

        logger.info(f"=== Создаём XLSX для даты {date_str} (до файла №{end_idx+1}) ===")

        # ---- ДОБАВЛЕНА ЛОГИКА test_days ----
        if test_days:
            start_idx = max(min_prev_files, len(cut_docs) - test_days)
        else:
            start_idx = min_prev_files

        cut_docs_effective = cut_docs[start_idx:]
        # ------------------------------------

        all_results = pd.DataFrame()

        for max_prev in range(3, 31):
            df = backtest_for_docs(cut_docs_effective, cache, quotes_df, max_prev)

            if df.empty:
                continue

            df = df.rename(columns={'cumulative': f'max_{max_prev}'})

            if all_results.empty:
                all_results = df
            else:
                all_results = all_results.merge(df, on='test_date', how='outer')

        if not all_results.empty:
            out_file = Path(xlsx_path / f"{date_str}.xlsx")
            all_results.to_excel(out_file, index=False, engine='openpyxl')
            logger.info(f"✔️ Создан файл: {out_file}")
        else:
            logger.warning(f"⚠️ Нет результатов для даты {date_str}")


if __name__ == '__main__':
    main()
