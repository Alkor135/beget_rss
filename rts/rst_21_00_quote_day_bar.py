"""
Отображение дневных котировок из базы данных в виде DataFrame с финансовым результатом за день.
Дневные котировки переделаны из минутных свечек с 21:00 МСК предыдущей торговой сессии.
"""

import pandas as pd
from pathlib import Path
import sqlite3

def prepare(path_db_quote: Path) -> pd.DataFrame:
    """
    Читает таблицу Futures из базы данных котировок и возвращает DataFrame.
    """
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT * FROM Futures", conn)
    # df = df.tail(20)
    df = df.sort_values('TRADEDATE', ascending=True)
    # print(df)

    df['next_bar'] = df.apply(
        lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1
    ).shift(-1)
    df['next_bar_pips'] = df.apply(
        lambda x: (x['CLOSE'] - x['OPEN']), axis=1
    ).shift(-1)

    df.columns = df.columns.str.lower()
    return df

if __name__ == '__main__':
    ticker = 'RTS'
    path_db_quote = Path(fr'C:\Users\Alkor\gd\data_quote_db\{ticker}_futures_day_2025_21-00.db')

    df = prepare(path_db_quote)
    df = df.sort_values('tradedate', ascending=True)
    print(df.tail(10).to_string(max_rows=30, max_cols=15))
