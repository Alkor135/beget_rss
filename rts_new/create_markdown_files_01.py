import pandas as pd
from pathlib import Path
import sqlite3
import logging
import yaml
from datetime import datetime

SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings['ticker']
ticker_lc = ticker.lower()
num_mds = settings['num_mds']
num_dbs = settings['num_dbs']
time_start = settings['time_start']
time_end = settings['time_end']
path_db_day = Path(settings['path_db_day'].replace('{ticker}', ticker))
db_news_dir = Path(settings['db_news_dir'])
md_path = Path(settings['md_path'])

(Path(__file__).parent / 'log').mkdir(parents=True, exist_ok=True)
log_file = Path(__file__).parent / 'log' / 'create_markdown_files.txt'

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

def read_news_dbs_to_df(db_dir: Path, num_dbs: int | None = None) -> pd.DataFrame:
    """
    Читает несколько файлов SQLite БД с новостями из директории db_dir
    в один DataFrame и сортирует по полю loaded_at.

    Ожидаемый формат файлов: rss_news_YYYY_MM.db
    Колонки в таблице: loaded_at, date, title, provider.
    """
    db_files = sorted(
        db_dir.glob("rss_news_*.db")
    )

    if num_dbs is not None and num_dbs > 0:
        db_files = db_files[-num_dbs:]  # последние num_dbs файлов

    all_rows = []

    for db_file in db_files:
        try:
            with sqlite3.connect(db_file) as conn:
                df_part = pd.read_sql_query(
                    "SELECT loaded_at, date, title, provider FROM news",
                    conn
                )
                df_part["source_db"] = db_file.name  # опционально: откуда строка
                all_rows.append(df_part)
            logging.info(f"Успешно прочитан файл БД: {db_file}")
        except Exception as e:
            logging.error(f"Ошибка чтения БД {db_file}: {e}")

    if not all_rows:
        logging.warning("Не удалось прочитать ни одного файла БД новостей")
        return pd.DataFrame(columns=["loaded_at", "date", "title", "provider", "source_db"])

    df_all = pd.concat(all_rows, ignore_index=True)

    # Приводим loaded_at к datetime и сортируем
    df_all["loaded_at"] = pd.to_datetime(df_all["loaded_at"])
    df_all = df_all.sort_values("loaded_at").reset_index(drop=True)

    return df_all

def build_trade_intervals(
    db_path: str,
    time_start: str,
    time_end: str,
    table_name: str = "Futures"
):
    """
    Читает отсортированную колонку TRADEDATE из SQLite-БД и строит интервалы:
    (prev_date + time_start, curr_date + time_end).

    Пример результата:
    (
        (datetime(2025, 6, 2, 21, 0), datetime(2025, 6, 3, 20, 59, 59)),
        (datetime(2025, 6, 3, 21, 0), datetime(2025, 6, 4, 20, 59, 59)),
        ...
    )
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT TRADEDATE FROM {table_name} ORDER BY TRADEDATE")
        rows = cur.fetchall()

    # Берём только список дат (str)
    dates = [r[0] for r in rows]

    # Нужно минимум две даты для построения хотя бы одного интервала
    if len(dates) < 2:
        return tuple()

    intervals = []

    for prev_date_str, curr_date_str in zip(dates[:-1], dates[1:]):
        # Склеиваем дату и время и переводим в datetime
        start_dt = datetime.fromisoformat(f"{prev_date_str} {time_start}")
        end_dt = datetime.fromisoformat(f"{curr_date_str} {time_end}")
        intervals.append((start_dt, end_dt))

    return tuple(intervals)

def create_markdown_files_from_intervals(
    df_news: pd.DataFrame,
    intervals: tuple,
    md_dir: Path,
    ticker: str,
) -> None:
    """
    По каждому интервалу (start_dt, end_dt) из intervals
    создаёт markdown-файл с заголовками новостей из df_news.title,
    у которых loaded_at попадает в этот интервал.

    Имя файла: YYYY-MM-DD.md, где дата берётся из end_dt элемента пары.
    """
    md_dir.mkdir(parents=True, exist_ok=True)

    # Убедимся, что loaded_at в datetime
    if not pd.api.types.is_datetime64_any_dtype(df_news["loaded_at"]):
        df_news = df_news.copy()
        df_news["loaded_at"] = pd.to_datetime(df_news["loaded_at"])

    for start_dt, end_dt in intervals:
        # Фильтрация новостей по интервалу
        mask = (df_news["loaded_at"] >= start_dt) & (df_news["loaded_at"] <= end_dt)
        df_slice = df_news.loc[mask].sort_values("loaded_at")

        if df_slice.empty:
            continue  # нет новостей — файл не создаём

        # Имя файла по дате конца интервала (первый элемент по твоему описанию)
        date_str = end_dt.date().isoformat()
        filename = f"{date_str}.md"
        filepath = md_dir / filename

        # Структура markdown — удобная для эмбеддингов:
        # заголовок файла (дата/тикер), затем одна новость в блоке:
        # время, провайдер и заголовок.
        lines = []
        # lines.append(f"# Новости {ticker} за интервал {start_dt} — {end_dt}")
        # lines.append("")
        for _, row in df_slice.iterrows():
            # news_time = row["loaded_at"]
            title = str(row["title"])
            # provider = str(row.get("provider", "") or "")
            # Один логический блок на новость
            # lines.append(f"## {news_time} — {provider}")
            # lines.append("")
            lines.append(title)
            lines.append("")  # пустая строка-разделитель

        content = "\n".join(lines)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Создан markdown-файл: {filepath}")
        except Exception as e:
            logging.error(f"Ошибка записи markdown-файла {filepath}: {e}")

if __name__ == "__main__":
    df_news = read_news_dbs_to_df(db_news_dir, num_dbs=num_dbs)
    with pd.option_context(  # Печать широкого и длинного датафрейма
            "display.width", 1000,
            "display.max_columns", 30,
            "display.max_colwidth", 100
    ):
        print("Датафрейм с результатом:")
        print(df_news)

    intervals = build_trade_intervals(
        db_path=path_db_day,  # из settings.yaml
        time_start=time_start,
        time_end=time_end,
        table_name="Futures"
    )
    for it in intervals[:5]:
        print(it)

    create_markdown_files_from_intervals(
        df_news=df_news,
        intervals=intervals[-num_mds:],  # например, только последние num_mds интервалов
        md_dir=md_path,
        ticker=ticker,
    )
    print(f"\nMarkdown файлы созданы в {md_path}")