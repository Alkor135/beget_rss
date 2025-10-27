import pickle
import pandas as pd
import numpy as np

CACHE_FILE = "rts_embeddings_investing_ollama.pkl"

# Загрузка кэша
with open(CACHE_FILE, "rb") as f:
    cache_data = pickle.load(f)

print(f"Всего записей в кэше: {len(cache_data)}")

# Превращаем в DataFrame
df = pd.DataFrame(cache_data)

print(df)
print(df.columns)
print(df['metadata'])
print(df['metadata'][0])