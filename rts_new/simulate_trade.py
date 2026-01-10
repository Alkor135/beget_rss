"""
Скрипт загружает дневные котировки фьючерсов и эмбеддинги новостей из БД и кэша, объединяя их по дате.
Для каждой даты начиная с заданной точки сравнивает вектор новости с предыдущими k векторами
(от 3 до 30) через косинусное сходство.
Определяет наиболее похожий день и сравнивает направление движения цены (по NEXT_BODY) в текущем и
найденном прошлом дне.
Если направления совпадают, записывает модуль доходности, иначе — минус её модуль (ошибка прогноза).
Считает накопленный P/L за test_days дней вперёд для каждого k и выбирает лучшее окно (MAX_k) по
максимальному P/L.
Формирует сигнал: использует значение MAX_k из лучшего окна как прогноз доходности на следующий день.
Выводит кумулятивный P/L по этим сигналам и строит график эффективности стратегии.
"""

from pathlib import Path
from datetime import datetime
import pickle
import sqlite3
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
ticker_lc = ticker.lower()
cache_file = Path(settings['cache_file'].replace('{ticker_lc}', ticker_lc))  # Путь к pkl-файлу с кэшем
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))  # Путь к БД дневных котировок
min_prev_files = settings.get('min_prev_files', 2)
test_days = settings.get('test_days', 23) + 1
START_DATE = settings.get('start_date', "2025-10-01")
# START_DT = datetime.strptime(START_DATE, "%Y-%m-%d").date()

# === Логирование ===
log_dir = Path(__file__).parent / 'log'
log_dir.mkdir(parents=True, exist_ok=True)
# Имя файла лога с датой и временем запуска (один файл на запуск!)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f'simulate_trade_{timestamp}.txt'

# Настройка логирования: ТОЛЬКО один файл + консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # один файл
        logging.StreamHandler()                           # консоль
    ]
)

# Ручная очистка старых логов (оставляем только 3 самых новых)
def cleanup_old_logs(log_dir: Path, max_files: int = 3):
    """Удаляет старые лог-файлы, оставляя max_files самых новых."""
    log_files = sorted(log_dir.glob("simulate_trade_*.txt"))
    if len(log_files) > max_files:
        for old_file in log_files[:-max_files]:
            try:
                old_file.unlink()
                print(f"Удалён старый лог: {old_file.name}")
            except Exception as e:
                print(f"Не удалось удалить {old_file}: {e}")

# Вызываем очистку ПЕРЕД началом логирования
cleanup_old_logs(log_dir, max_files=3)
logging.info(f"🚀 Запуск скрипта. Лог-файл: {log_file}")

def load_quotes(path_db_quote):
    """Загрузка котировок и расчет NEXT_BODY."""
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query(
            "SELECT TRADEDATE, OPEN, CLOSE FROM Futures",
            conn,
            parse_dates=['TRADEDATE']  # <-- Преобразуем TRADEDATE в datetime
        )
    df = df.set_index('TRADEDATE').sort_index()
    df['NEXT_BODY'] = (df['CLOSE'] - df['OPEN']).shift(-1)
    df = df.dropna(subset=['NEXT_BODY'])
    return df[['NEXT_BODY']]

def load_cache(cache_file_path):
    """Загрузка кэша эмбеддингов."""
    with open(cache_file_path, 'rb') as f:
        df = pickle.load(f)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    return df.set_index('TRADEDATE').sort_index()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Сравнение по косинусному сходству"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def compute_max_k(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    k: int,
    col_vectors: str = "VECTORS",
    col_body: str = "NEXT_BODY"
) -> pd.Series:
    """
    Возвращает Series для колонки MAX_k
    """
    result = pd.Series(index=df.index, dtype=float)

    dates = df.index
    start_pos = dates.get_loc(start_date)

    for i in range(start_pos, len(df)):
        if i < k:
            continue

        vec_cur = df.iloc[i][col_vectors]
        body_cur = df.iloc[i][col_body]

        similarities = []
        indices = []

        for j in range(i - k, i):
            vec_prev = df.iloc[j][col_vectors]
            sim = cosine_similarity(vec_cur, vec_prev)
            similarities.append(sim)
            indices.append(j)

        # индекс самой похожей строки
        best_j = indices[int(np.argmax(similarities))]
        body_prev = df.iloc[best_j][col_body]

        if np.sign(body_cur) == np.sign(body_prev):
            result.iloc[i] = abs(body_cur)
        else:
            result.iloc[i] = -abs(body_cur)

    return result

def main(path_db_day, cache_file):
    df_bar = load_quotes(path_db_day)  # Загрузка DF с дневными котировками (с 21:00 пред. сессии)
    df_emb = load_cache(cache_file)  # Загрузка DF с векторами новостей

    # Объединение датафреймов по индексу TRADEDATE
    df_combined = df_bar.join(df_emb[['VECTORS']], how='inner')  # 'inner' — только общие даты

    # Генерация колонок MAX_3 … MAX_30
    start_date = pd.to_datetime(START_DATE)
    for k in range(3, 31):
        col_name = f"MAX_{k}"
        logging.info(f"📊 Расчёт {col_name}")
        df_combined[col_name] = compute_max_k(
            df=df_combined,
            start_date=start_date,
            k=k
        )

    # === Замена NaN на 0.0 во всех MAX_ колонках ===
    max_cols = [f"MAX_{k}" for k in range(3, 31)]
    df_combined[max_cols] = df_combined[max_cols].fillna(0.0)

    # === Расчёт PL_ колонок ===
    for k in range(3, 31):
        max_col = f"MAX_{k}"
        pl_col = f"PL_{k}"

        df_combined[pl_col] = (
            df_combined[max_col]
            .shift(1)  # исключаем текущую строку
            .rolling(window=test_days, min_periods=1)
            .sum()
        )

    # Отладочный вывод
    with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 120
    ):
        print(df_bar)
        print(df_emb)
        print(df_combined[["NEXT_BODY", "VECTORS"]])
        print(df_combined)

    # === Замена NaN на 0.0 во всех колонках ===
    df_combined = df_combined.fillna(0.0)

    # === ОСТАВИТЬ ТОЛЬКО НУЖНЫЕ КОЛОНКИ ===
    final_cols = [f"MAX_{k}" for k in range(3, 31)] + [f"PL_{k}" for k in range(3, 31)]
    df_combined = df_combined[final_cols].copy()

    # Опционально: сортировка по индексу (по дате)
    df_combined.sort_index(inplace=True)

    # Отладочный вывод
    with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 24,
        "display.max_colwidth", 120,
        "display.min_rows", 30
    ):
        print(df_combined[[f"PL_{k}" for k in range(3, 31)]])

    # ===============================
    # Формирование df_rez
    # ===============================

    pl_cols = [f"PL_{k}" for k in range(3, 31)]
    max_cols = [f"MAX_{k}" for k in range(3, 31)]

    rows = []

    for idx, row in df_combined.iterrows():
        trade_date = idx

        # максимальное значение среди PL_3 ... PL_30
        pl_values = row[pl_cols]
        pl_max = pl_values.max()

        pl_result = 0.0

        if pl_max > 0.0:
            # имя колонки с максимальным PL
            best_pl_col = pl_values.idxmax()  # например "PL_7"
            n = int(best_pl_col.split("_")[1])  # -> 7

            # соответствующая колонка MAX_n
            max_col = f"MAX_{n}"
            pl_result = row[max_col]

        rows.append({
            "TRADEDATE": trade_date,
            "P/L": pl_result
        })

    df_rez = pd.DataFrame(rows).set_index("TRADEDATE")

    # ===============================
    # Вывод df_rez в консоль
    # ===============================
    with pd.option_context(
            "display.width", 1000,
            "display.max_columns", 10,
            "display.max_colwidth", 120
    ):
        print(df_rez)

    # ===============================
    # График cumulative P/L
    # ===============================
    df_rez["CUM_P/L"] = df_rez["P/L"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df_rez.index, df_rez["CUM_P/L"])
    plt.title("Cumulative P/L")
    plt.xlabel("Date")
    plt.ylabel("P/L")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # Сохранение графика
    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / f'cumulative_pl_{timestamp}.png'
    plt.savefig(plot_path)
    logging.info(f"📊 График сохранён: {plot_path}")
    plt.close()  # Освобождаем память

if __name__ == "__main__":
    main(path_db_day, cache_file)