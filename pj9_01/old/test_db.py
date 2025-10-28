import sqlite3
from pathlib import Path

DBS = [
    r"C:/Users/Alkor/gd/data_quote_db/RTS_futures_minute_2025.db",
    r"C:/Users/Alkor/gd/data_quote_db/minutes_RTS_2025.sqlite",
]

TIME_CANDIDATES = ["datetime", "timestamp", "ts", "time", "date", "dt"]

def pick_time_col(cur, table):
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1].lower() for row in cur.fetchall()]
    for cand in TIME_CANDIDATES:
        if cand in cols:
            return cand
    return None

def print_head_tail(conn, table, key):
    cur = conn.cursor()
    print(f"\n=== {Path(conn.execute('PRAGMA database_list').fetchone()[2]).name} :: {table} ===")
    if key:
        asc = f'ORDER BY "{key}" ASC'
        desc = f'ORDER BY "{key}" DESC'
    else:
        asc = "ORDER BY rowid ASC"
        desc = "ORDER BY rowid DESC"
    for title, order in [("FIRST 5", asc), ("LAST 5", desc)]:
        print(f"\n-- {title}")
        cur.execute(f'SELECT * FROM "{table}" {order} LIMIT 5')
        rows = cur.fetchall()
        for r in rows:
            print(r)

for db in DBS:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY 1")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        key = pick_time_col(cur, t)
        print_head_tail(conn, t, key)
    conn.close()
