"""
ChatGPT-5
Конвертор минутных котировок в дневные бары.
Особенности:
- Поддерживает несколько файлов с минутными котировками (по годам).
- При повторном запуске НЕ перезаписывает всю дневную БД:
    * Берёт максимальную дату в дневной БД (если есть) и начинает обработку с этой даты (пересчитывая её)
      и далее добавляет все отсутствующие дневные бары.
    * Если дневной БД нет — создаёт её и заполняет начиная с первой доступной даты (у которой есть предыдущий день).
- Учитывает rollover — когда внутри дневного интервала встречаются несколько SECID, корректирует старую часть на gap.
Выбор фьючерса зависит от параметра ticker в файле settings.yaml.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import logging
import yaml
import glob
import os

# Попытка подключить tqdm (если установлен) — для визуального прогресса.
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # безопасно работать без tqdm

# Путь к settings.yaml (в той же папке, что и скрипт)
# SETTINGS_FILE = Path(__file__).parent / "settings_rts.yaml"
SETTINGS_FILE = Path(__file__).parent / "settings_mix.yaml"

# --- Чтение настроек ---
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# Параметры (ожидаются в settings.yaml)
ticker = settings['ticker']
ticker_lc = ticker.lower()
provider = settings.get('provider', 'local')
# path_db_minute может быть:
# - точный путь к файлу,
# - путь с шаблоном (например "C:/.../minutes_RTS_*.sqlite"),
# - путь к каталогу, из которого будут взяты файлы minutes_{ticker}_*.sqlite
path_db_minutes = settings.get('path_db_minute', 'C:/Users/Alkor/gd/data_quote_db')
path_db_day = Path(settings.get('path_db_day').replace('{ticker}', ticker))
time_start = settings.get('time_start', '21:00:00')  # начало сессии (предыдущая сессия)
time_end = settings.get('time_end', '20:59:59')    # конец сессии (текущая)
output_dir = Path(settings.get('output_dir', '.')).resolve()
log_file = output_dir / 'log' / f'{ticker_lc}_21_00_convert_minutes_to_days.txt'

# --- Настройка логирования ---
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []  # удалить старые обработчики (во избежание дублирования)
# Файловый обработчик (перезаписывает файл при запуске)
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# Консольный обработчик
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def safe_connect(path_db: Path, logger, timeout: int = 30) -> sqlite3.Connection:
    """
    Безопасное подключение к файлу SQLite:
    - Создаёт родительскую папку, если её нет.
    - Пытается создать пустой файл базы, если файла нет.
    - Подключается с таймаутом.
    Логирует подробности и вызывает sqlite3.OperationalError дальше, если не удалось.
    """
    # Убедимся, что у нас Path
    path_db = Path(path_db)
    # Развернуть ~ и сделать абсолютным (без жесткого resolve, чтобы не падать при несуществующих дисках)
    try:
        path_db = path_db.expanduser()
    except Exception:
        pass

    logger.info(f"Попытка подключения к дневной БД: {path_db}")

    parent = path_db.parent
    try:
        # Создаём родительскую папку (если её нет)
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Не удалось создать папку {parent}: {e}")

    # Попробуем создать файл, если его нет (touch)
    try:
        if not path_db.exists():
            # touch в try/except, на случай проблем с правами/доступом
            path_db.touch()
            logger.info(f"Создан пустой файл дневной БД: {path_db}")
    except PermissionError as e:
        logger.error(f"Нет прав на создание файла {path_db}: {e}")
    except OSError as e:
        logger.error(f"OS error при создании файла {path_db}: {e}")

    # Диагностика до попытки connect
    logger.info(f"Диагностика: parent.exists={parent.exists()}, parent.is_dir={parent.is_dir()}, file.exists={path_db.exists()}")

    # Попытка подключения
    try:
        conn = sqlite3.connect(str(path_db), timeout=timeout)
        logger.info(f"Успешно подключились к дневной БД: {path_db}")
        return conn
    except sqlite3.OperationalError as e:
        # Даем подробные подсказки — что проверить
        logger.error(f"sqlite3.OperationalError при попытке открыть {path_db}: {e}")
        logger.error("Возможные причины: 1) Неправильный путь 2) Нет прав на запись 3) Файл занят другим процессом 4) длинный путь (>260 символов на Windows).")
        # Дополнительно логируем атрибуты пути
        try:
            logger.error(f"Доп. инфо: abs_path={path_db.absolute()}, parent_perm_ok={os.access(parent, os.W_OK)}, file_perm_ok={path_db.exists() and os.access(path_db, os.W_OK)}")
        except Exception:
            pass
        # Пробуем дать подсказку пользователю (не меняем поведение — пробрасываем исключение)
        raise

# --------------------
# Вспомогательные функции
# --------------------
def discover_minute_db_files(path_spec: str, ticker: str) -> list:
    """
    Находит файлы минутной БД по переданному path_spec:
    - если path_spec указывает на файл — возвращает [that_file]
    - если содержит '*' — использует glob
    - если path_spec указывает на директорию — ищет файлы minutes_{ticker}_*.sqlite в ней
    Возвращает отсортированный список путей (по имени файла).
    """
    p = Path(path_spec)
    files = []
    if p.exists() and p.is_file():
        files = [p]
    elif '*' in path_spec:
        files = [Path(x) for x in sorted(glob.glob(path_spec))]
    elif p.exists() and p.is_dir():
        pattern = str(p / f"minutes_{ticker}_*.sqlite")
        files = [Path(x) for x in sorted(glob.glob(pattern))]
    else:
        # Последняя попытка: если указан путь без wildcard, но не существует, попробуем glob-образный вариант
        pattern = path_spec
        files = [Path(x) for x in sorted(glob.glob(pattern))]
    # Отфильтровать несуществующие (без ошибок)
    files = [f for f in files if f.exists() and f.is_file()]
    return files


def create_day_table_if_missing(conn: sqlite3.Connection) -> None:
    """Создаёт таблицу с дневными барами, если её нет."""
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Futures (
                TRADEDATE DATE PRIMARY KEY UNIQUE NOT NULL,
                OPEN REAL NOT NULL,
                LOW REAL NOT NULL,
                HIGH REAL NOT NULL,
                CLOSE REAL NOT NULL,
                SECID TEXT NOT NULL,
                LSTTRADE TEXT NOT NULL
            )
        ''')
        conn.commit()


def get_max_trade_date_from_day_db(conn: sqlite3.Connection) -> str:
    """Возвращает максимальную TRADEDATE (строкой 'YYYY-MM-DD') или None"""
    cur = conn.cursor()
    cur.execute("SELECT MAX(TRADEDATE) FROM Futures")
    r = cur.fetchone()
    cur.close()
    if r and r[0]:
        return r[0]
    return None


def read_unique_dates_from_minute_files(minute_files: list) -> list:
    """
    Считывает уникальные календарные даты (DATE части TRADEDATE) из всех minute DB файлов.
    Возвращает отсортированный список дат в формате YYYY-MM-DD по возрастанию.
    ВНИМАНИЕ: читается только DISTINCT DATE(TRADEDATE), а не все минуты (это экономит память).
    """
    dates_set = set()
    for f in minute_files:
        try:
            conn = sqlite3.connect(str(f))
            cur = conn.cursor()
            # Здесь предполагается, что в минутной БД тоже таблица называется Futures и в ней столбец TRADEDATE
            cur.execute("SELECT DISTINCT DATE(TRADEDATE) FROM Futures")
            rows = cur.fetchall()
            for row in rows:
                if row and row[0]:
                    dates_set.add(row[0])
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка чтения дат из {f}: {e}")
    dates = sorted(list(dates_set))
    return dates


def fetch_minute_rows_for_interval(minute_files: list, start_ts: str, end_ts: str) -> list:
    """
    Для указанного интервала 'YYYY-MM-DD HH:MM:SS' - 'YYYY-MM-DD HH:MM:SS' собирает
    все минуты из всех minute_files и возвращает список строк:
    (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE)
    Отсортирован по TRADEDATE ASC.
    """
    combined = []
    for f in minute_files:
        try:
            conn = sqlite3.connect(str(f))
            cur = conn.cursor()
            cur.execute("""
                SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE
                FROM Futures
                WHERE TRADEDATE >= ? AND TRADEDATE <= ?
                ORDER BY TRADEDATE ASC
            """, (start_ts, end_ts))
            rows = cur.fetchall()
            if rows:
                combined.extend(rows)
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Ошибка выборки минут из {f} для {start_ts} - {end_ts}: {e}")
    # Сортировка на случай, если из разных файлов пришли пересекающиеся фрагменты
    combined.sort(key=lambda r: r[0])
    return combined


def build_daily_candle_from_minutes(rows: list, time_end: str) -> tuple:
    """
    Построение дневной свечки из списка минутных строк (rows).
    rows: список кортежей (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE), отсортированных по TRADEDATE ASC.
    time_end нужен только чтобы извлечь дату для TRADEDATE результата.
    Возвращает (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE) или None, если недостаточно данных.
    """
    if not rows:
        return None

    # Закрытие — CLOSE последней минуты
    last_row = rows[-1]
    # TRADEDATE результата — дата из last_row (только YYYY-MM-DD часть)
    try:
        trade_date = datetime.strptime(last_row[0], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
    except Exception:
        # Если формат иной, пробуем только взять первые 10 символов
        trade_date = last_row[0][:10]

    close = last_row[4]
    secid_last = last_row[5]
    lsttrade = last_row[6]

    # OPEN — OPEN первой минуты в интервале (по времени)
    open_overall = rows[0][1]

    # LOW и HIGH по умолчанию из всех минут
    lows = [r[2] for r in rows if r[2] is not None]
    highs = [r[3] for r in rows if r[3] is not None]
    if not lows or not highs:
        return None
    low_all = min(lows)
    high_all = max(highs)

    # Проверка на rollover — сколько уникальных SECID в интервале
    secids = []
    secid_positions = {}  # secid -> [indexes]
    for i, r in enumerate(rows):
        s = r[5]
        if s not in secid_positions:
            secid_positions[s] = []
            secids.append(s)
        secid_positions[s].append(i)

    if len(secids) <= 1:
        # Обычный случай — один контракт
        return (trade_date, open_overall, low_all, high_all, close, secid_last, lsttrade)
    else:
        # Rollover: найдём старый и новый контракт относительно последнего SECID
        new_secid = secid_last
        # индекс первой строки с новым SECID
        first_new_idx = secid_positions[new_secid][0]
        # последние индексы старого секцида — возьмём все предыдущие строки до first_new_idx
        # Найдём last_close_old (последняя строка перед сменой)
        if first_new_idx == 0:
            # вся секция — новая, нет старой части
            # Тогда работаем как с одним контрактом (новым)
            low_new = min([r[2] for r in rows if r[5] == new_secid])
            high_new = max([r[3] for r in rows if r[5] == new_secid])
            open_new = next((r[1] for r in rows if r[5] == new_secid), None)
            if open_new is None:
                return None
            return (trade_date, open_new, low_new, high_new, close, secid_last, lsttrade)

        # Последняя строка старого контракта:
        last_old_row = rows[first_new_idx - 1]
        last_close_old = last_old_row[4]

        # Первая строка нового контракта:
        first_new_row = rows[first_new_idx]
        first_open_new = first_new_row[1]

        # Вычисляем gap (корректировку)
        gap = first_open_new - last_close_old

        # Старые части: open_old = OPEN первой минуты старого сегмента; low_old/high_old — аггрегаты
        old_segment = [r for r in rows[:first_new_idx]]
        if not old_segment:
            # на случай пустого старого сегмента (защита)
            low_new = min([r[2] for r in rows if r[5] == new_secid])
            high_new = max([r[3] for r in rows if r[5] == new_secid])
            open_new = next((r[1] for r in rows if r[5] == new_secid), None)
            return (trade_date, open_new, low_new, high_new, close, secid_last, lsttrade)

        open_old = old_segment[0][1]
        low_old = min([r[2] for r in old_segment])
        high_old = max([r[3] for r in old_segment])

        # Корректируем старую часть на gap
        adj_open_old = open_old + gap
        adj_low_old = low_old + gap
        adj_high_old = high_old + gap

        # Новая часть (без корректировки)
        new_segment = [r for r in rows[first_new_idx:]]
        if not new_segment:
            # если вдруг нет новой части, используем скорректированные старые значения
            overall_low = adj_low_old
            overall_high = adj_high_old
            overall_open = adj_open_old
        else:
            low_new = min([r[2] for r in new_segment])
            high_new = max([r[3] for r in new_segment])
            overall_low = min(adj_low_old, low_new)
            overall_high = max(adj_high_old, high_new)
            overall_open = adj_open_old  # открытие берём скорректированное из старой части для непрерывности

        return (trade_date, overall_open, overall_low, overall_high, close, secid_last, lsttrade)


def save_daily_candle_if_missing(conn: sqlite3.Connection, candle: tuple) -> None:
    """
    Сохраняет дневную свечку в дневной БД, если записи для этой TRADEDATE ещё нет.
    candle: (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE)
    """
    if candle is None:
        return
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM Futures WHERE TRADEDATE = ?", (candle[0],))
    if cur.fetchone()[0] > 0:
        logger.info(f"Запись для {candle[0]} уже существует — пропускаем.")
        cur.close()
        return
    try:
        with conn:
            conn.execute("""
                INSERT INTO Futures (TRADEDATE, OPEN, LOW, HIGH, CLOSE, SECID, LSTTRADE)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, candle)
            conn.commit()
            logger.info(f"Добавлена дневная свечка для {candle[0]}")
    except Exception as e:
        logger.error(f"Ошибка при вставке дневной свечки для {candle[0]}: {e}")
    finally:
        cur.close()


# --------------------
# Главная логика
# --------------------
def main():
    logger.info("Запуск конвертора минут -> дни")
    minute_files = discover_minute_db_files(path_db_minutes, ticker)
    if not minute_files:
        logger.error(f"Не найдено ни одного файла минутной БД по шаблону: {path_db_minutes}")
        return

    logger.info(f"Найдено файлов минутной БД: {len(minute_files)}")
    for f in minute_files:
        logger.info(f" - {f}")

    # Считываем все уникальные даты из минутных файлов (YYYY-MM-DD), отсортированные по возрастанию
    dates = read_unique_dates_from_minute_files(minute_files)
    if not dates:
        logger.error("В минутных базах нет ни одной даты.")
        return
    logger.info(f"Найдено уникальных календарных дат в минутных данных: {len(dates)} (от {dates[0]} до {dates[-1]})")

    # Открываем/создаём дневную БД
    # conn_day = sqlite3.connect(str(path_db_day))

    # Проверка на существование дневной БД (отладочный код)
    try:
        conn_day = safe_connect(path_db_day, logger)
    except sqlite3.OperationalError as err:
        # Если не удалось подключиться — логируем и аккуратно завершаем.
        logger.error(f"Не удалось открыть/создать дневную БД по пути {path_db_day}: {err}")
        logger.error("Проверьте существование пути, права доступа и что файл не заблокирован другим процессом.")
        return

    create_day_table_if_missing(conn_day)

    # Определяем с какой даты начинать обработку:
    max_day = get_max_trade_date_from_day_db(conn_day)  # строка YYYY-MM-DD или None
    logger.info(f"Максимальная дата в дневной БД: {max_day}")

    # Находим индекс даты в списке dates для старта обработки.
    # Логика:
    # - если max_day существует в списке дат минут — начнём с индекса max(1, idx_of_max_day)
    #   чтобы пересчитать последний уже записанный день (в случае добора минутных данных),
    #   но если max_day — самая первая дата (idx==0), то пропустим (нет предыдущей даты для интервала).
    # - если max_day нет — начнём с индекса 1 (первый возможный дневной бар имеет previous date + current date),
    #   потому что для формирования дневного бара нужна предыдущая календарная дата.
    if max_day and max_day in dates:
        idx = dates.index(max_day)
        start_idx = max(1, idx)  # пересчитаем max_day (если есть prev day)
    else:
        # Нету max_day в minutes (или в дневной БД нет записей) — стартуем с индекса 1
        start_idx = 1

    # Собираем список дат, для которых будем формировать дневные бары:
    process_dates = dates[start_idx:]  # каждая such date имеет previous = dates[i-1]
    logger.info(f"Будут обработаны даты (начиная с индекса {start_idx}): {len(process_dates)}")

    # Итерация по датам с прогресс-баром
    iterator = process_dates
    if tqdm:
        iterator = tqdm(process_dates, desc="Processing days", unit="day")
    for end_date in iterator:
        # предыдущая дата в списке
        prev_idx = dates.index(end_date) - 1
        if prev_idx < 0:
            # нет предыдущего дня — пропускаем
            logger.info(f"Нет предыдущей календарной даты для {end_date}, пропускаем.")
            continue
        prev_date = dates[prev_idx]

        start_ts = f"{prev_date} {time_start}"
        end_ts = f"{end_date} {time_end}"

        # Получаем минуты из всех минутных файлов
        minute_rows = fetch_minute_rows_for_interval(minute_files, start_ts, end_ts)
        if not minute_rows:
            logger.info(f"Нет минутных данных за период {start_ts} - {end_ts}, пропускаем.")
            continue

        # Формируем дневной бар из собранных минут
        candle = build_daily_candle_from_minutes(minute_rows, time_end)
        if not candle:
            logger.info(f"Не удалось построить дневной бар для {end_date}")
            continue

        # Сохраняем если отсутствует
        save_daily_candle_if_missing(conn_day, candle)

    # Закрываем дневную БД
    conn_day.close()
    logger.info("Готово. Соединение с дневной БД закрыто.")


if __name__ == '__main__':
    main()
