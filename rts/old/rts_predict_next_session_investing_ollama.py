"""
Скрипт для предсказания направления следующей свечи (next_bar) на основе markdown-файлов с новостями.
Кэширует эмбеддинги в pickle-файл, проверяет актуальность кэша.
Ограничивает количество предыдущих файлов параметрами min_prev_files и max_prev_files.
Предсказывает направление для документа с next_bar='None'.
Все выводы print сохраняются в текстовый файл в папке predict_ollama с именем {none_date}.txt.
"""

from pathlib import Path
import pickle
import hashlib
import numpy as np
import yaml
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from contextlib import redirect_stdout
import datetime

# Параметры
ticker_lc = 'rts'
md_path = Path(fr'C:\Users\Alkor\gd\md_{ticker_lc}_investing')
cache_file = Path(fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\{ticker_lc}_embeddings_investing_ollama.pkl')
model_name = "bge-m3"
url_ai = "http://localhost:11434/api/embeddings"
min_prev_files = 4   # Минимальное количество предыдущих файлов для предсказания
max_prev_files = 7  # Максимальное количество предыдущих файлов для предсказания
# Папка для сохранения текстовых файлов
output_dir = Path(fr'C:\Users\Alkor\gd\predict_ai\{ticker_lc}_investing_ollama')

def cosine_similarity(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def load_markdown_files(directory):
    """Загружает все MD-файлы из директории, сортирует по дате (имени файла)."""
    files = sorted(directory.glob("*.md"), key=lambda f: f.stem)  # Сортировка по дате ascending
    documents = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1].strip()
                text_content = parts[2].strip()
                metadata = yaml.safe_load(metadata_yaml) or {}
                metadata_str = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),
                    "date_min": str(metadata.get("date_min", "unknown")),
                    "date_max": str(metadata.get("date_max", "unknown")),
                    "source": file_path.name,
                    "date": file_path.stem
                }
                doc = Document(page_content=text_content, metadata=metadata_str)
                documents.append(doc)
            else:
                doc = Document(
                    page_content=content,
                    metadata={
                        "next_bar": "unknown",
                        "date_min": "unknown",
                        "date_max": "unknown",
                        "source": file_path.name,
                        "date": file_path.stem
                    }
                )
                documents.append(doc)
        else:
            doc = Document(
                page_content=content,
                metadata={
                    "next_bar": "unknown",
                    "date_min": "unknown",
                    "date_max": "unknown",
                    "source": file_path.name,
                    "date": file_path.stem
                }
            )
            documents.append(doc)
    return documents

def cache_is_valid(documents, cache_file):
    """Проверяет, актуален ли кэш эмбеддингов."""
    if not cache_file.exists():
        return False

    cache_mtime = cache_file.stat().st_mtime
    current_files = {doc.metadata['source'] for doc in documents}

    # Загружаем кэш для проверки
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)

    cache_files = {item['metadata']['source'] for item in cache}

    # Проверяем, совпадают ли наборы файлов
    if current_files != cache_files:
        print("Кэш устарел: изменился набор markdown-файлов.")
        return False

    # Проверяем, не изменились ли файлы
    for doc in documents:
        file_path = md_path / doc.metadata['source']
        if file_path.stat().st_mtime > cache_mtime:
            print(f"Кэш устарел: файл {file_path.name} был изменён.")
            return False

    return True

def cache_embeddings(documents, cache_file, model_name, url_ai):
    """Вычисляет и кэширует эмбеддинги всех документов в pickle-файл."""
    if cache_is_valid(documents, cache_file):
        print(f"Загрузка кэша эмбеддингов из {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        return cache

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print(f"{timestamp} - Вычисление эмбеддингов...")
    ef = OllamaEmbeddingFunction(model_name=model_name, url=url_ai)
    cache = []
    for doc in documents:
        embedding = ef([doc.page_content])[0]
        cache.append({
            'id': hashlib.md5(doc.page_content.encode()).hexdigest(),
            'embedding': embedding,
            'metadata': doc.metadata
        })
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print(f"{timestamp} - Эмбеддинги сохранены в {cache_file}")
    return cache

def main(max_prev_files: int = 8):
    """Предсказывает направление next_bar для документа с next_bar='None'."""
    # Загрузка markdown-файлов
    documents = load_markdown_files(md_path)
    if len(documents) < min_prev_files + 1:
        print(f"Недостаточно файлов: {len(documents)}. Требуется минимум {min_prev_files + 1}.")
        exit(1)

    # Кэширование эмбеддингов
    cache = cache_embeddings(documents, cache_file, model_name, url_ai)

    # Создание папки для вывода, если она не существует
    output_dir.mkdir(parents=True, exist_ok=True)

    # Поиск документа с next_bar="None"
    none_doc = None
    for doc in documents:
        if doc.metadata['next_bar'] == "None":
            none_doc = doc
            break

    if not none_doc:
        print("Документ с next_bar='None' не найден.")
        return

    none_date = none_doc.metadata['date']
    output_file = output_dir / f"{none_date}.txt"

    # Перенаправление вывода в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print(f"\nПредсказание для даты {none_date} (next_bar='None'), {max_prev_files=}:")

            # Получение эмбеддинга документа с next_bar="None"
            none_id = hashlib.md5(none_doc.page_content.encode()).hexdigest()
            none_embedding = None
            for item in cache:
                if item['id'] == none_id:
                    none_embedding = item['embedding']
                    break
            if none_embedding is None:
                print(f"Эмбеддинг для даты {none_date} не найден в кэше.")
                return

            # Получение предыдущих документов из кэша, ближайших по дате
            prev_cache = sorted(
                [item for item in cache if item['metadata']['next_bar'] != "None" and item['metadata']['date'] < none_date],
                key=lambda x: x['metadata']['date'], reverse=True
            )[:max_prev_files]  # Ограничиваем max_prev_files ближайшими датами

            if len(prev_cache) < min_prev_files:
                print(f"Недостаточно предыдущих документов для даты {none_date}: {len(prev_cache)}")
                return

            # Вычисление сходств
            similarities = [
                (cosine_similarity(none_embedding, item['embedding']) * 100, item['metadata'])
                for item in prev_cache
            ]

            # Сортировка по убыванию сходства
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Ближайший документ
            if similarities:
                closest_similarity, closest_metadata = similarities[0]
                predicted_next_bar = closest_metadata['next_bar']

                print(f"\nПредсказание для даты {none_date} {min_prev_files}/{max_prev_files}:")
                print(f"Предсказанное направление: {predicted_next_bar}")
                print(f"Процент сходства: {closest_similarity:.2f}%")
                print("Метаданные ближайшего похожего документа:")
                for key in sorted(closest_metadata.keys()):
                    print(f"  {key}: {closest_metadata[key]}")
            else:
                print("Нет похожих документов для предсказания.")

    print(f"Результаты сохранены в {output_file}")
    print(f"\nПредсказание для даты {none_date}:")
    print(f"Предсказанное направление для {min_prev_files}/{max_prev_files}: {predicted_next_bar}")

if __name__ == '__main__':
    main(max_prev_files=max_prev_files)