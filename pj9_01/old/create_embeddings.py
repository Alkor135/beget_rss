import sqlite3
import pandas as pd
import requests
import json
from pathlib import Path
from tqdm import tqdm
import hashlib
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Параметры
db_files = [
    r"C:\Users\Alkor\gd\db_rss\rss_news_2025_09.db",
    r"C:\Users\Alkor\gd\db_rss\rss_news_2025_10.db",
]
table_name = "news"
url_ai = "http://localhost:11434/api/embeddings"
model_name = "bge-m3"

# Итоговый файл и файл частичного прогресса (чекпоинт)
final_pkl = "news_embeds.pkl"
checkpoint_pkl = "partial.pkl"

# Размер порции для чекпоинтов
chunk_size = 200

def order_fingerprint(df: pd.DataFrame) -> str:
    # Фингерпринт порядка и идентичности записи по ключевым полям
    # Берём loaded_at (ns), title, provider, date(ns)
    s = (
        df["loaded_at"].astype("int64").astype(str).fillna("NA")
        + "||" + df["title"].astype(str)
        + "||" + df["provider"].astype(str)
        + "||" + df["date"].astype("int64").astype(str).fillna("NA")
    )
    h = hashlib.sha256("||".join(s.tolist()).encode("utf-8")).hexdigest()
    return h

def load_base_df() -> pd.DataFrame:
    # 1) Считать и объединить таблицы из всех БД с прогресс‑баром по файлам
    dfs = []
    for path in tqdm(db_files, desc="Загрузка SQLite файлов"):
        con = sqlite3.connect(path)
        try:
            df_part = pd.read_sql_query(
                f"SELECT loaded_at, date, title, provider FROM {table_name}",
                con
            )
            df_part["__source_db__"] = path
            dfs.append(df_part)
        finally:
            con.close()
    if not dfs:
        raise RuntimeError("Нет данных из БД.")
    df = pd.concat(dfs, ignore_index=True)

    # 2) Преобразование строк в datetime
    for col in ["loaded_at", "date"]:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # 3) Детерминированная сортировка
    df = df.sort_values(by=["loaded_at", "title"], kind="mergesort").reset_index(drop=True)

    # 4) Разница во времени и фильтр < 5 минут
    df["load_delay"] = df["loaded_at"] - df["date"]
    df = df.loc[df["load_delay"].notna() & (df["load_delay"] < pd.Timedelta(minutes=5))].copy()

    # Финальный reset_index после фильтра
    df.reset_index(drop=True, inplace=True)

    # 5) Подготовка колонки под эмбеддинги как object, заполнение None
    df["title_embedding"] = pd.Series([None] * len(df), dtype=object)
    return df

def post_embed(session: requests.Session, text: str, timeout=60):
    payload = {"model": model_name, "input": text}
    resp = session.post(
        url_ai,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding")
    if vec is None:
        raise RuntimeError(f"Нет поля 'embedding' в ответе: {data}")
    return vec

def save_checkpoint(df_partial: pd.DataFrame, fp: str):
    # Сохраняем и метаданные порядка
    meta = {
        "_meta": {
            "total_len": int(len(df_partial)),
            "order_fp": fp,
        }
    }
    # Храним в один pickle: мету — в атрибуты через .attrs у DataFrame
    df_to_save = df_partial.copy()
    df_to_save.attrs.update(meta["_meta"])
    Path(checkpoint_pkl).parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_pickle(checkpoint_pkl)

def load_checkpoint():
    p = Path(checkpoint_pkl)
    if not p.exists():
        return None, None
    dfp = pd.read_pickle(p)
    # Извлекаем attrs, если доступны
    total_len = dfp.attrs.get("total_len", None)
    order_fp = dfp.attrs.get("order_fp", None)
    return dfp, {"total_len": total_len, "order_fp": order_fp}

def main():
    # База (объединение, сортировка, фильтр, подготовка колонки)
    base_df = load_base_df()
    base_fp = order_fingerprint(base_df)

    # Восстанавливаем прогресс
    partial_df, meta = load_checkpoint()
    if partial_df is not None and meta is not None:
        same_fp = (meta.get("order_fp") == base_fp)
        usable = same_fp and (len(partial_df) <= len(base_df))
    else:
        usable = False

    if usable:
        # Копируем базовую и переносим готовые эмбеддинги по индексу
        df = base_df.copy()
        if "title_embedding" in partial_df.columns:
            n = len(partial_df)
            # Приводим столбец к object, если вдруг нет
            if df["title_embedding"].dtype != object:
                df["title_embedding"] = df["title_embedding"].astype(object)
            # Поэлементная запись: безопасно для разнородных списков
            for i in range(n):
                df.at[i, "title_embedding"] = partial_df.at[i, "title_embedding"]
        else:
            df = base_df.copy()
    else:
        df = base_df.copy()

    # Сколько уже готово
    done_mask = df["title_embedding"].notna()
    done_count = int(done_mask.sum())

    session = requests.Session()

    # Основной цикл по порциям
    with tqdm(total=len(df), desc="Эмбеддинги (Ollama)", unit="row") as pbar:
        pbar.update(done_count)
        start = done_count
        while start < len(df):
            end = min(start + chunk_size, len(df))
            # Поэлементная обработка с безопасной записью
            for idx in range(start, end):
                title = "" if pd.isna(df.at[idx, "title"]) else str(df.at[idx, "title"])
                vec = post_embed(session, title)
                # Гарантированная запись по одному элементу в object-столбец
                df.at[idx, "title_embedding"] = vec
                pbar.update(1)

            # Чекпоинт: сохраняем df[:end] и fingerprint текущего базового df
            save_checkpoint(df.iloc[:end], base_fp)
            start = end

    # Финальная запись и удаление чекпоинта
    Path(final_pkl).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(final_pkl)
    try:
        Path(checkpoint_pkl).unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Готово. Строк: {len(df)}. Файл: {final_pkl}")

if __name__ == "__main__":
    main()
