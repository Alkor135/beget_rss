import pandas as pd
from pathlib import Path
import sqlite3

def rts_prepare(path_db_quote: Path) -> pd.DataFrame:
    """
    Читает таблицу Futures из базы данных котировок и возвращает DataFrame.
    """
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT * FROM Futures", conn)
    df = df.tail(20)

    df['rts_next_bar'] = df.apply(
        lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1
    ).shift(-1)
    df['rts_next_bar_pips'] = df.apply(
        lambda x: (x['CLOSE'] - x['OPEN']), axis=1
    ).shift(-1)

    df.columns = df.columns.str.lower()
    df.rename(columns={'secid': 'rts_secid', 'open': 'rts_open', 'close': 'rts_close'}, inplace=True)
    df.drop(['low', 'high', 'lsttrade'], axis=1, inplace=True)
    return df

def mix_prepare(path_db_quote: Path) -> pd.DataFrame:
    """
    Читает таблицу Futures из базы данных котировок и возвращает DataFrame.
    """
    with sqlite3.connect(path_db_quote) as conn:
        df = pd.read_sql_query("SELECT * FROM Futures", conn)
    df = df.tail(20)

    df['mix_next_bar'] = df.apply(
        lambda x: 'up' if (x['OPEN'] < x['CLOSE']) else 'down', axis=1
    ).shift(-1)
    df['mix_next_bar_pips'] = df.apply(
        lambda x: (x['CLOSE'] - x['OPEN']), axis=1
    ).shift(-1)

    df.columns = df.columns.str.lower()
    df.rename(columns={'secid': 'mix_secid', 'open': 'mix_open', 'close': 'mix_close'}, inplace=True)
    df.drop(['low', 'high', 'lsttrade'], axis=1, inplace=True)
    return df

if __name__ == '__main__':
    rts_path_db_quote = Path(fr'c:\Users\Alkor\gd\data_beget_rss\RTS_day_rss_2025.db')
    mix_path_db_quote = Path(fr'c:\Users\Alkor\gd\data_beget_rss\MIX_day_rss_2025.db')

    df_rts = rts_prepare(rts_path_db_quote)
    df_mix = mix_prepare(mix_path_db_quote)

    # Правое слияние по полю 'tradedate'
    df = pd.merge(df_rts, df_mix, on='tradedate', how='right')

    # pd.set_option('display.max_colwidth', 10)
    # print(df)
    print(df.to_string(max_rows=30, max_cols=15))
