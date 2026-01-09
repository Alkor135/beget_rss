"""
Скрипт загружает кэшированные данные из `.pkl`-файла, содержащего, вероятно, embeddings и метаданные.
Данные десериализуются с помощью библиотеки `pickle`.
Затем они преобразуются в `DataFrame` для удобного анализа.
Выводится общее количество записей в кэше.
Отображаются названия столбцов: `embedding`, `metadata` и, возможно, другие.
Пример содержимого первого элемента метаданных показывает структуру источника.
Скрипт полезен для отладки и анализа кэшированных векторных представлений текстов.
"""

import pickle
import pandas as pd

CACHE_FILE = "rts_embeddings_investing_ollama.pkl"

# Загрузка кэша
with open(CACHE_FILE, "rb") as f:
    cache_data = pickle.load(f)

print(f"Всего записей в кэше: {len(cache_data)}")
print()

# Превращаем в DataFrame
df = pd.DataFrame(cache_data)

print(df)
print()
print(df.columns)
print()
print(df['metadata'])
print()
print(df['metadata'][0])
print()
print(df['embedding'])