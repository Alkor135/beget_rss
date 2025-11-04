import os
import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Конфигурация ---
PKL_PATH = r"combine.pkl"                 # путь к входному pkl-файлу
OUTPUT_XLSX = r"result_similar_byday.xlsx" # путь к выходному Excel-файлу
PRINT_MAX_ROWS = 50                        # ограничение печати в консоль
CHECKPOINT_EVERY = 200                     # периодичность чекпоинтов (в строках)
CHECKPOINT_DIR = r".checkpoints_sim_byday" # директория для чекпоинтов

# --- Вспомогательные функции ---

def ensure_datetime(df, col):
    """Преобразовать столбец в datetime при необходимости."""
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def cosine_sim_matrix(A, B):
    """Косинусное сходство между матрицами A (n_a x d) и B (n_b x d)."""
    eps = 1e-9
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A_norm @ B_norm.T

# --- Загрузка данных ---
print("Loading:", PKL_PATH)
df = pd.read_pickle(PKL_PATH)

# Удаляем все строки, где есть NaN в любых столбцах (по требованию)
before_rows = len(df)
df = df.dropna(axis=0, how='any').reset_index(drop=True)
after_rows = len(df)
print(f"Dropped rows with NaN: {before_rows - after_rows}")

# Проверяем наличие обязательных столбцов
required = [
    'loaded_at','date','title','provider','embedding',
    'TRADEDATE','OPEN','CLOSE','H2','perc_25','perc_75'
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Гарантируем корректный тип datetime и добавляем столбец date_only (только дата)
df = ensure_datetime(df, 'date')
if df['date'].isna().any():
    bad = df[df['date'].isna()]
    raise ValueError(f"Некорректные значения в 'date'. Count={len(bad)}")

df['date_only'] = df['date'].dt.date

# Преобразуем эмбеддинги в ndarray одинаковой размерности
def to_ndarray(x):
    return np.asarray(x, dtype=float)

emb = df['embedding'].map(to_ndarray)
lens = emb.map(lambda a: a.shape[0] if a.ndim==1 else (a.shape[1] if a.ndim>1 else 0))
if lens.nunique() != 1:
    raise ValueError("Embeddings have inconsistent dimensions")

df = df.assign(emb_vec=emb)

# Сортируем по времени для корректной хронологии
df = df.sort_values('date').reset_index(drop=True)

# Индексация записей по дням для быстрого доступа к предыдущим дням
by_day = {}
for idx, row in df.iterrows():
    d = row['date_only']
    by_day.setdefault(d, []).append(idx)

unique_days = sorted(by_day.keys())
day_pos = {d:i for i,d in enumerate(unique_days)}

# Контейнеры для результатов
rows_out = []              # индексы строк, прошедших правило
flag_map = {}              # индекс -> флаг ('all_low' или 'all_high')
intermediate_records = []  # необязательная промежуточная информация (для отладки)

# Матрица эмбеддингов
E = np.vstack(df['emb_vec'].to_list())

# Подготовка директории для чекпоинтов
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
ckpt_result_path = os.path.join(CHECKPOINT_DIR, 'result_partial.parquet')
ckpt_intermediate_path = os.path.join(CHECKPOINT_DIR, 'intermediate_partial.parquet')

# --- Основной цикл с прогресс-баром ---
N = len(df)
for cur_idx, cur_row in tqdm(df.iterrows(), total=N, desc='Processing rows', unit='row'):
    cur_day = cur_row['date_only']
    pos = day_pos[cur_day]
    if pos == 0:
        # Для самого первого дня предыдущих дней нет
        continue

    # Собираем кандидатов из всех предыдущих дней
    prev_indices = []
    for d in unique_days[:pos]:
        prev_indices.extend(by_day[d])
    if not prev_indices:
        continue

    # Косинусные сходства текущего эмбеддинга со всеми предыдущими
    cur_vec = E[cur_idx:cur_idx+1]
    prev_mat = E[prev_indices]
    sims = cosine_sim_matrix(cur_vec, prev_mat).ravel()

    # Для каждого предыдущего дня выбираем одну лучшую запись, затем берём глобально топ-3
    day_to_best = {}
    for j, idx_prev in enumerate(prev_indices):
        d = df.at[idx_prev, 'date_only']
        s = sims[j]
        if d not in day_to_best or s > day_to_best[d][1]:
            day_to_best[d] = (idx_prev, s)

    per_day_best_list = list(day_to_best.values())
    per_day_best_list.sort(key=lambda x: x[1], reverse=True)
    top3 = per_day_best_list[:3]

    # Сохраняем промежуточную информацию (необязательно)
    for idx_prev, sim in top3:
        intermediate_records.append({
            'current_index': cur_idx,
            'current_date': cur_row['date'],
            'current_title': cur_row.get('title', None),
            'match_index': idx_prev,
            'match_date': df.at[idx_prev, 'date'],
            'match_title': df.at[idx_prev, 'title'] if 'title' in df.columns else None,
            'similarity': sim,
            'match_H2': df.at[idx_prev, 'H2'],
            'match_perc_25': df.at[idx_prev, 'perc_25'],
            'match_perc_75': df.at[idx_prev, 'perc_75'],
        })

    # Применяем правило: все три ниже perc_25 или все три выше perc_75
    if len(top3) == 3:
        lows = [df.at[idx_prev, 'H2'] < df.at[idx_prev, 'perc_25'] for idx_prev, _ in top3]
        highs = [df.at[idx_prev, 'H2'] > df.at[idx_prev, 'perc_75'] for idx_prev, _ in top3]
        cond_low = all(lows)
        cond_high = all(highs)
        if cond_low or cond_high:
            rows_out.append(cur_idx)
            flag_map[cur_idx] = 'all_low' if cond_low else 'all_high'

    # Периодические чекпоинты (для долгих прогонов)
    if (cur_idx + 1) % CHECKPOINT_EVERY == 0:
        try:
            pd.DataFrame(intermediate_records).to_parquet(ckpt_intermediate_path, index=False)
        except Exception:
            pd.DataFrame(intermediate_records).to_csv(ckpt_intermediate_path.replace('.parquet','.csv'), index=False)
        try:
            df_ckpt = df.loc[rows_out].copy()
            df_ckpt['similarity_flag'] = [flag_map[i] for i in rows_out]
            # Расчёт rez и rez_cum также кладём в чекпоинт
            def calc_rez(h2, flag):
                # Правило: модуль H2, если (H2>0 и all_high) или (H2<0 и all_low);
                # иначе -|H2|
                abs_h2 = abs(h2)
                if (h2 > 0 and flag == 'all_high') or (h2 < 0 and flag == 'all_low'):
                    return abs_h2
                else:
                    return -abs_h2
            df_ckpt['rez'] = [calc_rez(h2, fl) for h2, fl in zip(df_ckpt['H2'], df_ckpt['similarity_flag'])]
            df_ckpt['rez_cum'] = df_ckpt['rez'].cumsum()
            df_ckpt.to_parquet(ckpt_result_path, index=False)
        except Exception:
            df_ckpt = df.loc[rows_out].copy()
            df_ckpt['similarity_flag'] = [flag_map[i] for i in rows_out]
            def calc_rez(h2, flag):
                abs_h2 = abs(h2)
                if (h2 > 0 and flag == 'all_high') or (h2 < 0 and flag == 'all_low'):
                    return abs_h2
                else:
                    return -abs_h2
            df_ckpt['rez'] = [calc_rez(h2, fl) for h2, fl in zip(df_ckpt['H2'], df_ckpt['similarity_flag'])]
            df_ckpt['rez_cum'] = df_ckpt['rez'].cumsum()
            df_ckpt.to_csv(ckpt_result_path.replace('.parquet','.csv'), index=False)

# --- Финальная сборка результата ---
result_df = df.loc[rows_out].copy().reset_index(drop=True)
result_df['similarity_flag'] = [flag_map[i] for i in rows_out]

# Добавляем столбец rez по заданному правилу и кумулятивный rez_cum
# Правило для rez:
#   rez = |H2|, если (H2 > 0 и similarity_flag == 'all_high') или (H2 < 0 и similarity_flag == 'all_low')
#   rez = -|H2|, если (H2 > 0 и similarity_flag == 'all_low')  или (H2 < 0 и similarity_flag == 'all_high')
def calc_rez(h2, flag):
    abs_h2 = abs(h2)
    if (h2 > 0 and flag == 'all_high') or (h2 < 0 and flag == 'all_low'):
        return abs_h2
    else:
        return -abs_h2

result_df['rez'] = [calc_rez(h2, fl) for h2, fl in zip(result_df['H2'], result_df['similarity_flag'])]
result_df['rez_cum'] = result_df['rez'].cumsum()

# --- Вывод и сохранение ---
pd.set_option('display.max_rows', PRINT_MAX_ROWS)
pd.set_option('display.width', 160)
print("\nResult DataFrame (first {} rows):".format(PRINT_MAX_ROWS))
print(result_df.head(PRINT_MAX_ROWS))

result_df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nSaved result to: {OUTPUT_XLSX}")
