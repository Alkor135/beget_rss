import pandas as pd
from pathlib import Path
import sqlite3
import logging
import yaml

SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings['ticker']
ticker_lc = ticker.lower()
num_mds = settings['num_mds']
num_dbs = settings['num_dbs']
time_start = settings['time_start']
time_end = settings['time_end']
path_db_day = settings['path_db_day']
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

# Пример использования:
if __name__ == "__main__":
    df_news = read_news_dbs_to_df(db_news_dir, num_dbs=num_dbs)
    with pd.option_context(  # Печать широкого и длинного датафрейма
            "display.width", 1000,
            "display.max_columns", 30,
            "display.max_colwidth", 100
    ):
        print("Датафрейм с результатом:")
        print(df_news)
