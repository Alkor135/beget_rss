"""
Отображение котировок из базы данных в виде DataFrame для RTS.
Дневные котировки переделаны из минутных свечек с 21:00 МСК.
"""

import pandas as pd
from pathlib import Path
import sqlite3

def rts_prepare(path_db_quote: Path) -> pd.DataFrame:
    """
    Читает таблицу Futures из базы данных котировок и возвращает DataFrame.
    """
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT * FROM Futures", conn)
    # df = df.tail(20)
    df = df.sort_values('TRADEDATE', ascending=True)
    # print(df)

    df['rts_next_bar'] = df.apply(
        lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1
    ).shift(-1)
    df['rts_next_bar_pips'] = df.apply(
        lambda x: (x['CLOSE'] - x['OPEN']), axis=1
    ).shift(-1)

    df.columns = df.columns.str.lower()
    # df.rename(columns={'secid': 'rts_secid', 'open': 'rts_open', 'close': 'rts_close'}, inplace=True)
    # df.drop(columns=['low', 'high', 'lsttrade'], inplace=True, errors='ignore')
    return df

if __name__ == '__main__':
    rts_path_db_quote = Path(fr'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_2025_21-00.db')

    df_rts = rts_prepare(rts_path_db_quote)

    df_rts = df_rts.sort_values('tradedate', ascending=True)

    # print(df_rts.to_string(max_rows=30, max_cols=15))

    print(df_rts.tail(30).to_string(max_rows=30, max_cols=15))

    # print(df_rts)
