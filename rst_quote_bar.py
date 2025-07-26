import pandas as pd
from pathlib import Path
import sqlite3


if __name__ == '__main__':
    ticker = 'RTS'
    path_db_quote = Path(fr'c:\Users\Alkor\gd\data_beget_rss\{ticker}_day_rss_2025.db')

    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT * FROM Futures", conn)

    df = df.tail(30)
    df['next_bar'] = df.apply(
        lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1
    ).shift(-1)
    df['next_bar_pips'] = df.apply(
        lambda x: (x['CLOSE'] - x['OPEN']), axis=1
    ).shift(-1)

    # pd.set_option('display.max_colwidth', 10)
    # print(df)
    print(df.to_string(max_rows=30, max_cols=10))
