"""
Скрипт для чтения базы данных котировок фьючерсов RTS и отображения интерактивного свечного графика.
"""
import sqlite3
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

def main(path_db_day: Path) -> None:
    """
    Основная функция для чтения базы данных котировок и отображения графика.
    """
    # Подключение к базе данных SQLite
    conn = sqlite3.connect(path_db_day)

    # Чтение данных из таблицы в DataFrame с сортировкой по TRADEDATE
    df = pd.read_sql('SELECT * FROM Futures ORDER BY TRADEDATE ASC', conn)

    # Закрытие соединения
    conn.close()

    # Переименование колонок в нижний регистр
    df.columns = df.columns.str.lower()

    # Переименование колонки 'tradedate' в 'datetime'
    df = df.rename(columns={'tradedate': 'datetime'})

    # Преобразование столбца datetime в формат datetime64 и удаление времени
    df['datetime'] = pd.to_datetime(df['datetime']).dt.date

    # Вывод последних 30 строк DataFrame
    print(df.tail(25).to_string(max_rows=30, max_cols=15))

    # Проверка наличия необходимых колонок
    required_columns = {'datetime', 'open', 'close', 'high', 'low'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame должен содержать колонки: {required_columns}")

    # Убедитесь, что числовые столбцы имеют правильный тип
    df[['open', 'close', 'high', 'low']] = df[['open', 'close', 'high', 'low']].astype(float)

    # Создание свечного графика
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='blue',  # Синий для восходящих свечей
        decreasing_line_color='red'   # Красный для нисходящих свечей
    )])

    # Настройка макета графика
    fig.update_layout(
        title='RTS Futures',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,  # Отключение ползунка диапазона
        xaxis_type='category',  # Исключение пропущенных дат (выходных)
        xaxis_tickformat='%d.%m.%Y',  # Формат даты, например, 01.08.2025
        height=800  # Увеличение высоты графика (в пикселях)
    )

    # Отображение графика
    fig.show()

if __name__ == '__main__':
    path_db_day = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_2025_21-00.db')
    main(path_db_day)