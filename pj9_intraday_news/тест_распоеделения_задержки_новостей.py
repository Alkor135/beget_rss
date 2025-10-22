import sqlite3
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')  # интерактивный бэкэнд для PyCharm
import matplotlib.pyplot as plt
from pathlib import Path

# Ограничение по lag для графика (в минутах)
MAX_LAG_MINUTES = 20

def main():
    # Папка с БД
    db_folder = Path(r"C:\Users\Alkor\gd\db_rss")

    # Автоматический поиск всех файлов rss_news_*.db
    db_files = sorted(db_folder.glob("rss_news_*.db"))

    if not db_files:
        raise FileNotFoundError("Файлы rss_news_*.db не найдены в папке.")

    # Список для хранения DataFrame
    dfs = []

    # Чтение данных из каждой БД
    for db_file in db_files:
        print(f"Загружаем данные из {db_file}")
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT loaded_at, date, title, provider FROM news", conn)
        conn.close()
        dfs.append(df)

    # Объединяем все DataFrame
    df_all = pd.concat(dfs, ignore_index=True)

    # Преобразуем строки в datetime
    df_all['loaded_at'] = pd.to_datetime(df_all['loaded_at'])
    df_all['date'] = pd.to_datetime(df_all['date'])

    # Добавляем колонку 'lag' в минутах
    df_all['lag'] = (df_all['loaded_at'] - df_all['date']).dt.total_seconds() / 60

    # Статистика
    print("Статистика lag (мин):")
    print(df_all['lag'].describe())

    # Ограничение лагов для графика
    df_plot = df_all[df_all['lag'] <= MAX_LAG_MINUTES]

    # Построение графика распределения
    plt.figure(figsize=(10,6))
    plt.hist(df_plot['lag'], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Распределение lag новостей (в минутах, max {MAX_LAG_MINUTES})')
    plt.xlabel('Lag (минуты)')
    plt.ylabel('Количество новостей')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig("lag_distribution.png", dpi=150)  # сохраняем в PNG
    print("График сохранён в lag_distribution.png")


if __name__ == "__main__":
    main()
