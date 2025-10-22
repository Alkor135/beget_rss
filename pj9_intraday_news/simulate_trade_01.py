#!/usr/bin/env python3
"""
simulate_trade.py — версия с FAISS (по календарным датам)

Правила:
- Для строки с loaded_at = 2025-10-05 ищем похожие эмбеддинги только среди записей,
  где loaded_at.date <= 2025-10-04 (строго предыдущие календарные дни, включительно).
- Используем FAISS IndexFlatIP на нормализованных эмбеддингах (inner product == cosine для L2-нормированных векторов).
- Индекс накапливаем по датам: перед обработкой даты D индекс содержит все эмбеддинги для дат <= D-1.
- Требуется faiss (pip install faiss-cpu). Для больших наборов данных это даёт существенное ускорение.

Как работает скрипт:
1. Загружает кэш (pickle).
2. Подготавливает DataFrame: parsed loaded_at, date (календарная), float H2/Percentile, embedding -> np.float32.
3. Нормализует эмбеддинги (L2) и сохраняет embedding_norm.
4. Группирует по дате и итеративно:
   - добавляет в FAISS эмбеддинги предыдущей даты(ей);
   - обрабатывает все записи текущей даты (если в индексе >= TOP_N выполняется поиск).
5. Сохраняет результаты в Excel и применяет фильтр "минимум 2 часа между сделками".

Комментарий: скрипт ориентирован на большие наборы (100k+). Если faiss отсутствует, бросается понятная ошибка.
"""

import pickle
from pathlib import Path
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

# Попытка импортировать faiss
try:
    import faiss
except Exception as e:
    raise ImportError(
        "FAISS не найден. Установите faiss-cpu: pip install faiss-cpu\n"
        f"Ошибка импорта: {e}"
    )

# ----------------------
# Настройки (можно править)
# ----------------------
CACHE_FILE = "news_h2.pkl"
START_DATE = "2025-09-28"  # дата начала теста (включая)
TOP_N = 3  # количество ближайших похожих
PERCENTILE_THRESHOLDS = [0.9, 0.9, 0.9]  # пороги по перцентилю для топ-3
OUTPUT_FILE = "simulate_trade_results_faiss.xlsx"
EMBED_DIM = 1024  # размерность эмбеддингов bge-m3 (как было указано)
# ----------------------


def load_cache(cache_file: str):
    """Загрузить pickle с кэшем эмбеддингов."""
    if not Path(cache_file).exists():
        raise FileNotFoundError(f"Файл кэша не найден: {cache_file}")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data


def prepare_df(cache_data: list) -> pd.DataFrame:
    df = pd.DataFrame(cache_data)

    df["loaded_at"] = pd.to_datetime(df["metadata"].apply(lambda x: x["loaded_at"]))
    df["date"] = df["loaded_at"].dt.date

    df["H2"] = df["metadata"].apply(lambda x: float(x["H2"]) if x.get("H2") is not None else np.nan)
    df["Percentile"] = df["metadata"].apply(
        lambda x: float(x["Percentile"]) if x.get("Percentile") is not None else np.nan
    )

    # Преобразуем embedding в numpy-массив (если None или не список — заменим на пустой массив)
    def safe_convert(x):
        if x is None:
            return np.array([], dtype=np.float32)
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.array(x, dtype=np.float32)
        return np.array([], dtype=np.float32)

    df["embedding"] = df["embedding"].apply(safe_convert)

    # Фильтрация испорченных / пустых эмбеддингов
    before = len(df)
    df = df[df["embedding"].apply(lambda x: len(x) == EMBED_DIM)]
    removed = before - len(df)
    print(f"⚠️ Удалено {removed} записей с некорректным embedding (None, пустой или размерности != {EMBED_DIM})")

    # Фильтрация по H2 и Percentile
    before = len(df)
    df = df.dropna(subset=["H2", "Percentile"]).reset_index(drop=True)
    removed2 = before - len(df)
    print(f"⚠️ Удалено {removed2} записей с None в H2 или Percentile")

    # Нормализация L2
    def l2_normalize(vec):
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    df["embedding_norm"] = df["embedding"].apply(l2_normalize)

    df = df.sort_values("loaded_at").reset_index(drop=True)
    return df


def simulate_with_faiss(df: pd.DataFrame, start_date: str):
    """
    Симуляция с использованием FAISS.
    Алгоритм:
    - группируем df по датам (календарно)
    - проходим по датам в порядке возрастания
    - перед обработкой даты D индекс должен содержать все записи с date <= D-1
      (то есть мы добавляем эмбеддинги предыдущей даты(ей) в индекс по ходу итерации)
    """
    start_ts = pd.to_datetime(start_date)

    # Список уникальных дат в отсортированном порядке
    unique_dates = np.array(sorted(df["date"].unique()))

    # FAISS: IndexFlatIP (inner product). Работает на float32.
    d = EMBED_DIM
    index = faiss.IndexFlatIP(d)  # inner product индекс
    # Вспомогательные списки метаданных, которые соответствуют порядку в FAISS (индексы из search -> эти списки)
    index_H2 = []
    index_percentile = []
    index_loaded_at = []

    results = []

    # Проход по датам
    # Для i==0 индекс пуст (нет прошлых дат) -> обработка будет пропускать, если нет достаточного количества прошлых записей
    for i, cur_date in enumerate(tqdm(unique_dates, desc="Проход по датам (FAISS)")):
        # Добавляем в индекс все эмбеддинги для предыдущей даты (если i > 0)
        if i > 0:
            prev_date = unique_dates[i - 1]
            df_prev = df[df["date"] == prev_date]
            if len(df_prev) > 0:
                # Стек нормализованных эмбеддингов
                embs = np.stack(df_prev["embedding_norm"].to_numpy()).astype(np.float32)
                # Добавляем в FAISS
                index.add(embs)
                # И добавляем соответствующие метаданные в параллельные списки
                index_H2.extend(df_prev["H2"].tolist())
                index_percentile.extend(df_prev["Percentile"].tolist())
                index_loaded_at.extend(df_prev["loaded_at"].tolist())

        # Обрабатываем все строки текущей даты (если date >= START_DATE)
        # Под условием: loaded_at >= start_ts
        df_cur = df[(df["date"] == cur_date) & (df["loaded_at"] >= start_ts)]
        if df_cur.empty:
            continue

        # Если в индексе меньше TOP_N — пропускаем обработку записей этой даты,
        # т.к. недостаточно прошлых эмбеддингов
        if index.ntotal < TOP_N:
            # Не забываем: записи текущей даты всё равно попадут в индекс позже, когда пройдём следующую дату
            # (по логике "добавляем prev_date перед обработкой cur_date")
            # Просто пропускаем обработку — нет достаточного количества прошлых примеров
            continue

        # Для ускорения: можно выполнить пакетный поиск всех эмбеддингов текущей даты в FAISS
        # Соберём Xq: массив (N_cur, dim)
        Xq = np.stack(df_cur["embedding_norm"].to_numpy()).astype(np.float32)
        k = TOP_N
        # Выполним поиск
        D, I = index.search(Xq, k)  # D: similarity (inner product), I: индексы в index
        # D.shape == (N_cur, k), I.shape == (N_cur, k)

        # Проходим по результатам поиска и применяем бизнес-логику
        for row_idx, (dist_row, idx_row) in enumerate(zip(D, I)):
            # idx_row содержит индексы в индексированных массивах (index_H2 и т.д.)
            # Получаем top_H2 и top_percentile в порядке убывания похожести (FAISS уже возвращает по убыванию)
            top_H2 = [index_H2[idx] for idx in idx_row]
            top_percentiles = [index_percentile[idx] for idx in idx_row]

            # Проверка порогов Percentile (в порядке от самого похожего к менее похожим)
            # Если хотя бы один None или nan — эти значения уже были отфильтрованы ранее в prepare_df
            if not (top_percentiles[0] > PERCENTILE_THRESHOLDS[0] and
                    top_percentiles[1] > PERCENTILE_THRESHOLDS[1] and
                    top_percentiles[2] > PERCENTILE_THRESHOLDS[2]):
                continue

            # Берём H2 текущей строки
            cur_row = df_cur.iloc[row_idx]  # т.к. Xq порядок такой же как df_cur
            cur_H2 = cur_row["H2"]

            # Логика определения направления (та же, что и в оригинале)
            # Если все три top_H2 > 0 => прогноз на повышение -> dir 'buy'
            if all(h > 0 for h in top_H2):
                if cur_H2 > 0:
                    results.append({"loaded_at": cur_row["loaded_at"], "H2": abs(cur_H2), "dir": "buy"})
                elif cur_H2 < 0:
                    results.append({"loaded_at": cur_row["loaded_at"], "H2": -abs(cur_H2), "dir": "buy"})
            # Если все три top_H2 < 0 => прогноз на понижение -> dir 'sell'
            elif all(h < 0 for h in top_H2):
                if cur_H2 < 0:
                    results.append({"loaded_at": cur_row["loaded_at"], "H2": abs(cur_H2), "dir": "sell"})
                elif cur_H2 > 0:
                    results.append({"loaded_at": cur_row["loaded_at"], "H2": -abs(cur_H2), "dir": "sell"})
            # иначе — не добавляем сделку

    # Превращаем в DataFrame и убираем близкие события (<2 часа)
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df["loaded_at"] = pd.to_datetime(result_df["loaded_at"])
    result_df = result_df.sort_values(by="loaded_at", ascending=True).reset_index(drop=True)

    # # Фильтрация: между соседними сделками >= 2 часа (оставляем первую и те, где diff >= 2 часа)
    # result_df["time_diff"] = result_df["loaded_at"].diff()
    # result_df = result_df[(result_df["time_diff"].isna()) | (result_df["time_diff"] >= pd.Timedelta(hours=2))]
    # result_df = result_df.drop(columns=["time_diff"]).reset_index(drop=True)

    return result_df


def main():
    print("Загрузка кэша эмбеддингов...")
    cache_data = load_cache(CACHE_FILE)  # Загрузка кэша из pickle
    print(f"Загружено записей из кэша: {len(cache_data)}")

    # Возьмём первый embedding из кэша ============================================================
    first_emb = None
    for item in cache_data:
        emb = item.get("embedding")
        if emb is not None and len(emb) > 0:
            first_emb = np.array(emb)
            break

    if first_emb is None:
        print("❌ В кэше вообще нет валидных эмбеддингов (все пустые или None)")
    else:
        print("✅ Найден первый валидный embedding:")
        print(" - Тип:", type(first_emb))
        print(" - Форма:", first_emb.shape)
        print(" - Первые 10 значений:", first_emb[:10])
    # ============================================================================================

    df = prepare_df(cache_data)  # Подготовка DataFrame
    print(f"После фильтрации (H2/Percentile) и подготовки: {len(df)} записей")

    print("Запуск симуляции с FAISS...")
    results_df = simulate_with_faiss(df, START_DATE)

    # Преобразуем столбец loaded_at в datetime
    results_df['loaded_at'] = pd.to_datetime(results_df['loaded_at'])
    # Оставляем строки с временем 09:00:00 и позже
    results_df = results_df[results_df['loaded_at'].dt.time >= pd.to_datetime('09:00:00').time()]
    results_df['H2_cumsum'] = results_df['H2'].cumsum()
    print(f"Сделок после фильтров: {len(results_df)}")

    # Сохранение результатов в Excel (если есть)
    if not results_df.empty:
        results_df.to_excel(OUTPUT_FILE, index=False)
        print(f"Результаты сохранены в {OUTPUT_FILE}")
    else:
        print("Результаты пусты — файл не создан.")


if __name__ == "__main__":
    main()
