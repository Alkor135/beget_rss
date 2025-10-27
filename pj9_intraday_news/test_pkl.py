import pickle
import pandas as pd
import numpy as np

CACHE_FILE = "news_h2.pkl"

# Загрузка кэша
with open(CACHE_FILE, "rb") as f:
    cache_data = pickle.load(f)

print(f"Всего записей в кэше: {len(cache_data)}")

# Превращаем в DataFrame
df = pd.DataFrame(cache_data)

# Приводим H2 и Percentile к float
df['H2'] = df['metadata'].apply(lambda x: float(x['H2']) if x['H2'] is not None else np.nan)
df['Percentile'] = df['metadata'].apply(lambda x: float(x['Percentile']) if x['Percentile'] is not None else np.nan)

# Дата загрузки новости
df['loaded_at'] = pd.to_datetime(df['metadata'].apply(lambda x: x['loaded_at']))

# Сортируем по дате
df = df.sort_values('loaded_at').reset_index(drop=True)

# Вывод первых 20 строк
print(df['embedding'].head(5))
print(df['embedding'].tail(5))

# Если нужно вывести все строки, можно использовать
# pd.set_option('display.max_rows', None)
print(df)
print(df.columns)
print(df['metadata'])
print(df['metadata'][0])
