"""
Скрипт генерирует эмбеддинги для заголовков новостей из SQLite-баз (rss_news_*.db) с помощью Ollama (bge-m3).
Поддерживает возобновление после сбоев через чекпоинты. Обрабатывает данные последовательно, фильтрует по задержке <5 мин.
Результат — файл news_embeds.pkl с колонкой embedding. Использует OllamaEmbeddingFunction из chromadb.
Автоматически объединяет несколько БД, обеспечивает детерминированную сортировку и контроль целостности.
"""
import sqlite3
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Используем OllamaEmbeddingFunction из chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ---------------------
# Параметры
# ---------------------
# Укажите одну или несколько папок и масок. Поддерживается рекурсивный поиск.
db_globs = [
    r"C:\Users\Alkor\gd\db_rss\rss_news_*.db",   # маска для всех месяцев
    # r"D:\backup\rss\rss_news_*.db",            # можно добавить дополнительные пути
]

table_name = "news"

# Настройки Ollama
ollama_host = "http://localhost:11434"
ollama_model = "bge-m3"  # замените при необходимости

# Итоговый файл и файл частичного прогресса (чекпоинт)
final_pkl = "news_embeds.pkl"
checkpoint_pkl = "partial.pkl"

# Размер порции для чекпоинтов
chunk_size = 200


# ---------------------
# Поиск файлов БД
# ---------------------
def list_db_files() -> list[str]:
    """
    Собирает все .db файлы по заданным маскам db_globs.
    - Поддерживает шаблоны вида 'C:\\path\\rss_news_*.db'
    - Не добавляет каталоги
    - Удаляет дубликаты
    - Стабильно сортирует список (по имени)
    """
    files = []
    for pattern in db_globs:
        p = Path(pattern)
        # Если паттерн включает поддиректории, Path.glob сам разберет.
        # Для рекурсии можно вместо glob использовать rglob со звездой в каталоге:
        # но здесь оставим стандартный glob по маске.
        parent = p.parent
        name = p.name
        # Если имя масочное, используем glob; если конкретный файл — проверяем его существование.
        if any(ch in name for ch in ["*", "?", "[", "]"]):
            for f in parent.glob(name):
                if f.is_file():
                    files.append(str(f))
        else:
            if p.is_file():
                files.append(str(p))

    # Удаляем дубликаты, сортируем
    files = sorted(set(files))
    if not files:
        raise RuntimeError("Не найдены файлы БД по заданным маскам db_globs.")
    return files


# ---------------------
# Вспомогательные функции
# ---------------------
def order_fingerprint(df: pd.DataFrame) -> str:
    """
    Детерминированный фингерпринт порядка и идентичности записи по ключевым полям.
    Используем loaded_at(ns), title, provider, date(ns).
    """
    for col in ["loaded_at", "date", "title", "provider"]:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df))

    loaded_ns = pd.to_datetime(df["loaded_at"], errors="coerce").astype("int64", errors="ignore")
    date_ns = pd.to_datetime(df["date"], errors="coerce").astype("int64", errors="ignore")

    s = (
        loaded_ns.astype(str).fillna("NA")
        + "||" + df["title"].astype(str)
        + "||" + df["provider"].astype(str)
        + "||" + date_ns.astype(str).fillna("NA")
    )
    h = hashlib.sha256("||".join(s.tolist()).encode("utf-8")).hexdigest()
    return h


def load_base_df() -> pd.DataFrame:
    """
    1) Находит все БД по маскам db_globs.
    2) Считывает и объединяет таблицы news.
    3) Преобразует строки в datetime.
    4) Сортирует детерминированно.
    5) Фильтрует: задержка < 5 минут.
    6) Создает колонку embedding и убирает лишнее.
    """
    db_files = list_db_files()

    dfs = []
    for path in tqdm(db_files, desc="Загрузка SQLite файлов"):
        con = sqlite3.connect(path)
        try:
            df_part = pd.read_sql_query(
                f"SELECT loaded_at, date, title, provider FROM {table_name}",
                con,
            )
            dfs.append(df_part)
        finally:
            con.close()

    if not dfs:
        raise RuntimeError("Нет данных из БД.")

    df = pd.concat(dfs, ignore_index=True)

    # Преобразование строк в datetime
    for col in ["loaded_at", "date"]:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Детерминированная сортировка
    df = df.sort_values(by=["loaded_at", "title"], kind="mergesort").reset_index(drop=True)

    # Разница во времени и фильтр < 5 минут
    df["load_delay"] = df["loaded_at"] - df["date"]
    df = df.loc[df["load_delay"].notna() & (df["load_delay"] < pd.Timedelta(minutes=5))].copy()

    # Финальный reset_index
    df.reset_index(drop=True, inplace=True)

    # Колонка с эмбеддингами
    df["embedding"] = pd.Series([None] * len(df), dtype=object)

    # Убрать временную колонку для экономии места
    if "load_delay" in df.columns:
        df.drop(columns=["load_delay"], inplace=True)

    # На всякий случай уберем служебные колонки, если попадут
    for c in ["__source_db__"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df


def save_checkpoint(df_partial: pd.DataFrame, fp: str):
    """
    Сохраняем частичный прогресс + метаданные порядка в attrs.
    """
    meta = {
        "_meta": {
            "total_len": int(len(df_partial)),
            "order_fp": fp,
        }
    }
    df_to_save = df_partial.copy()
    df_to_save.attrs.update(meta["_meta"])
    Path(checkpoint_pkl).parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_pickle(checkpoint_pkl)


def load_checkpoint():
    """
    Загружаем чекпоинт, возвращаем (df, meta) или (None, None).
    """
    p = Path(checkpoint_pkl)
    if not p.exists():
        return None, None
    dfp = pd.read_pickle(p)
    total_len = dfp.attrs.get("total_len", None)
    order_fp = dfp.attrs.get("order_fp", None)
    return dfp, {"total_len": total_len, "order_fp": order_fp}


# ---------------------
# Эмбеддинги через OllamaEmbeddingFunction
# ---------------------
def build_embedder() -> OllamaEmbeddingFunction:
    """
    Создаёт и возвращает OllamaEmbeddingFunction для заданной модели и хоста.
    """
    return OllamaEmbeddingFunction(
        model_name=ollama_model,
        url=f"{ollama_host}/api/embeddings",
    )


def embed_batch(embedder: OllamaEmbeddingFunction, texts: list[str]) -> list[list[float]]:
    """
    Обработчик батча строк в эмбеддинги.
    """
    return embedder(texts)


# ---------------------
# Основной сценарий
# ---------------------
def main():
    # Подготовка базы
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
        df = base_df.copy()
        # Поддержка старого названия колонки (title_embedding) для обратной совместимости
        if "embedding" in partial_df.columns or "title_embedding" in partial_df.columns:
            src_col = "embedding" if "embedding" in partial_df.columns else "title_embedding"
            n = len(partial_df)
            if df["embedding"].dtype != object:
                df["embedding"] = df["embedding"].astype(object)
            for i in range(n):
                df.at[i, "embedding"] = partial_df.at[i, src_col]
        else:
            df = base_df.copy()
    else:
        df = base_df.copy()

    # Сколько уже готово
    done_mask = df["embedding"].notna()
    done_count = int(done_mask.sum())

    # Создаём эмбеддер
    embedder = build_embedder()

    # Основной цикл по порциям
    with tqdm(total=len(df), desc="Эмбеддинги (OllamaEmbeddingFunction)", unit="row") as pbar:
        pbar.update(done_count)
        start = done_count

        while start < len(df):
            end = min(start + chunk_size, len(df))

            # Собираем тексты для текущей порции
            titles = []
            indices = []
            for idx in range(start, end):
                title = "" if pd.isna(df.at[idx, "title"]) else str(df.at[idx, "title"])
                titles.append(title)
                indices.append(idx)

            # Получаем эмбеддинги батчем
            vectors = embed_batch(embedder, titles)

            # Защитная проверка длины
            if len(vectors) != len(indices):
                raise RuntimeError(f"Длина эмбеддингов ({len(vectors)}) != длина батча ({len(indices)})")

            # Записываем по местам
            for i, idx in enumerate(indices):
                df.at[idx, "embedding"] = vectors[i]
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
