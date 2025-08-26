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
import aiohttp
import asyncio
from contextlib import redirect_stdout
import datetime
import json

# Параметры
ticker_lc = 'rts'
md_path = Path(fr'C:\Users\Alkor\gd\md_{ticker_lc}_investing')
cache_file = Path(
    fr'C:\Users\Alkor\PycharmProjects\beget_rss\{ticker_lc}\{ticker_lc}_embeddings_investing_ollama.pkl')
model_name = "bge-m3"
url_ai = "http://localhost:11434/api/embeddings"
min_prev_files = 4  # Минимальное количество предыдущих файлов для предсказания
max_prev_files = 7  # Максимальное количество предыдущих файлов для предсказания
# Папка для сохранения текстовых файлов
output_dir = Path(fr'C:\Users\Alkor\gd\predict_ai\{ticker_lc}_investing_ollama')
batch_size = 1  # Размер батча для параллельных запросов эмбеддингов


def cosine_similarity(vec1, vec2):
    """Оптимизированная версия вычисления косинусного сходства"""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


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

    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)

    cache_files = {item['metadata']['source'] for item in cache}

    if current_files != cache_files:
        print("Кэш устарел: изменился набор markdown-файлов.")
        return False

    for doc in documents:
        file_path = md_path / doc.metadata['source']
        if file_path.stat().st_mtime > cache_mtime:
            print(f"Кэш устарел: файл {file_path.name} был изменён.")
            return False

    return True


async def get_embedding(text, model_name, url):
    """Асинхронно получает эмбеддинг для текста через API Ollama."""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model_name,
            "prompt": text
        }
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("embedding")
                else:
                    print(f"Ошибка API для текста: {response.status}")
                    return None
        except Exception as e:
            print(f"Ошибка при запросе эмбеддинга: {e}")
            return None


async def get_embeddings_batch(texts, model_name, url):
    """Обрабатывает батч текстов асинхронно."""
    tasks = [get_embedding(text, model_name, url) for text in texts]
    return await asyncio.gather(*tasks)


async def cache_embeddings_async(documents, cache_file, model_name, url_ai):
    """Создание эмбеддингов для новых и измененных файлов markdown асинхронно."""
    current_files = {
        doc.metadata['source']: (doc, hashlib.md5(doc.page_content.encode()).hexdigest()) for doc
        in documents
    }

    cache = []
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        cache = [item for item in cache if
                 item['metadata']['source'] in current_files and item['id'] ==
                 current_files[item['metadata']['source']][1]]

    cached_sources = {item['metadata']['source'] for item in cache}
    new_docs = [doc for doc in documents if doc.metadata['source'] not in cached_sources]

    if new_docs:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        new_files = [doc.metadata['source'] for doc in new_docs]
        print(
            f"{timestamp} - Вычисление эмбеддингов для {len(new_docs)} "
            f"новых/изменённых файлов: {', '.join(new_files)}")

        contents = [doc.page_content for doc in new_docs]
        embeddings = []

        # Разделяем на батчи
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            batch_embeddings = await get_embeddings_batch(batch, model_name, url_ai)
            embeddings.extend(batch_embeddings)

        for i, doc in enumerate(new_docs):
            if embeddings[i] is not None:
                cache.append({
                    'id': hashlib.md5(doc.page_content.encode()).hexdigest(),
                    'embedding': embeddings[i],
                    'metadata': doc.metadata
                })
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        print(f"{timestamp} - Кэш обновлён в {cache_file}")

    return cache


def cache_embeddings(documents, cache_file, model_name, url_ai):
    """Обёртка для вызова асинхронной функции."""
    return asyncio.run(cache_embeddings_async(documents, cache_file, model_name, url_ai))


def main(max_prev_files: int = 7):
    """Предсказывает направление next_bar для документа с next_bar='None'."""
    documents = load_markdown_files(md_path)
    if len(documents) < min_prev_files + 1:
        print(f"Недостаточно файлов: {len(documents)}. Требуется минимум {min_prev_files + 1}.")
        exit(1)

    cache = cache_embeddings(documents, cache_file, model_name, url_ai)

    output_dir.mkdir(parents=True, exist_ok=True)

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
    output_file.unlink(missing_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print(f"\nПредсказание для даты {none_date} (next_bar='None'), {max_prev_files=}:")
            none_id = hashlib.md5(none_doc.page_content.encode()).hexdigest()
            none_embedding = None
            for item in cache:
                if item['id'] == none_id:
                    none_embedding = item['embedding']
                    break
            if none_embedding is None:
                print(f"Эмбеддинг для даты {none_date} не найден в кэше.")
                return

            prev_cache = sorted(
                [item for item in cache if
                 item['metadata']['next_bar'] != "None" and item['metadata']['date'] < none_date],
                key=lambda x: x['metadata']['date'], reverse=True
            )[:max_prev_files]

            if len(prev_cache) < min_prev_files:
                print(
                    f"Недостаточно предыдущих документов для даты {none_date}: {len(prev_cache)}")
                return

            prev_files = [item['metadata']['source'] for item in prev_cache]
            print(f"Файлы для сравнения: {', '.join(prev_files)}")

            similarities = [
                (cosine_similarity(none_embedding, item['embedding']) * 100, item['metadata'])
                for item in prev_cache
            ]

            similarities.sort(key=lambda x: x[0], reverse=True)

            if similarities:
                closest_similarity, closest_metadata = similarities[0]
                predicted_next_bar = closest_metadata['next_bar']
                print(
                    f"\nПредсказание для даты: {none_date}, "
                    f"с параметрами: {min_prev_files}/{max_prev_files}:")
                print(f"Предсказанное направление: {predicted_next_bar}")
                print(f"Процент сходства: {closest_similarity:.2f}%")
                print("Метаданные ближайшего похожего документа:")
                for key in sorted(closest_metadata.keys()):
                    print(f"  {key}: {closest_metadata[key]}")
            else:
                print("Нет похожих документов для предсказания.")

    print(f"Результаты сохранены в {output_file}")
    print(
        f"\nПредсказание для даты: {none_date}, с параметрами: {min_prev_files}/{max_prev_files}:")
    print(f"Файлы для сравнения: {', '.join(prev_files)}")
    print(f"Предсказанное направление: {predicted_next_bar}")


if __name__ == '__main__':
    main(max_prev_files=max_prev_files)