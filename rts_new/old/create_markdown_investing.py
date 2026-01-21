# Неверная логика создания

import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# --- настройки путей ---
DB_DIR = Path(r"C:\Users\Alkor\gd\db_rss_investing")
MD_DIR = Path(r"C:\Users\Alkor\gd\md_investing")

# --- базовая настройка логов ---
LOG_DIR = Path(__file__).parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = LOG_DIR / f"create_markdown_investing_{timestamp}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def cleanup_old_logs(log_dir: Path, max_files: int = 3) -> None:
    """Удаляет старые лог-файлы, оставляя max_files самых новых."""
    log_files = sorted(log_dir.glob("create_markdown_investing_*.txt"))
    if len(log_files) > max_files:
        for old_file in log_files[:-max_files]:
            try:
                old_file.unlink()
                print(f"Удалён старый лог: {old_file.name}")
            except Exception as e:
                print(f"Не удалось удалить {old_file}: {e}")

cleanup_old_logs(LOG_DIR, max_files=3)
logging.info(f"Запуск скрипта. Лог-файл: {log_file}")

def iter_news_rows_from_all_dbs(db_dir: Path):
    """
    Итерирует по всем строкам всех БД формата rss_news_investing_YYYY_MM.db.
    Ожидается таблица `news` с колонками: date, title.
    Возвращает кортежи (date_str, title_str).
    """
    db_files = sorted(db_dir.glob("rss_news_investing_*.db"))
    if not db_files:
        logging.warning(f"В каталоге {db_dir} не найдено файлов rss_news_investing_*.db")
        return

    for db_file in db_files:
        logging.info(f"Чтение БД: {db_file}")
        try:
            with sqlite3.connect(db_file) as conn:
                cur = conn.cursor()
                cur.execute("SELECT date, title FROM news ORDER BY date")
                for date_str, title in cur.fetchall():
                    yield date_str, title
        except Exception as e:
            logging.error(f"Ошибка чтения {db_file}: {e}")

def group_news_by_date(db_dir: Path):
    """
    Группирует новости по дате (YYYY-MM-DD).
    Возвращает dict: { 'YYYY-MM-DD': [title1, title2, ...], ... }
    """
    grouped: dict[str, list[str]] = {}
    for date_str, title in iter_news_rows_from_all_dbs(db_dir):
        if not date_str:
            continue
        # date_str ожидается вида 'YYYY-MM-DD HH:MM:SS'
        try:
            day = date_str.split(" ")[0]
        except Exception:
            # на всякий случай пробуем через datetime
            try:
                dt = datetime.fromisoformat(date_str)
                day = dt.date().isoformat()
            except Exception:
                logging.warning(f"Не удалось распарсить дату: {date_str}")
                continue

        grouped.setdefault(day, []).append(str(title) if title is not None else "")

    return grouped

def write_markdown_files(grouped_news: dict[str, list[str]], md_dir: Path) -> None:
    """
    Записывает markdown-файлы по дням.
    Имя файла: YYYY-MM-DD.md
    Содержимое: заголовки, разделённые пустой строкой.
    """
    md_dir.mkdir(parents=True, exist_ok=True)

    for day, titles in sorted(grouped_news.items()):
        filename = f"{day}.md"
        filepath = md_dir / filename

        lines: list[str] = []
        for t in titles:
            lines.append(t)
            lines.append("")  # пустая строка-разделитель

        content = "\n".join(lines)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Создан markdown-файл: {filepath}")
        except Exception as e:
            logging.error(f"Ошибка записи файла {filepath}: {e}")

if __name__ == "__main__":
    grouped = group_news_by_date(DB_DIR)
    logging.info(f"Найдено дней с новостями: {len(grouped)}")
    write_markdown_files(grouped, MD_DIR)
    print(f"Markdown-файлы созданы в {MD_DIR}")
