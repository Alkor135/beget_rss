from langchain_core.documents import Document
import os
import yaml
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import shutil
import hashlib
import numpy as np

# Пути и параметры
md_path = Path(r'C:\Users\Alkor\gd\md_rss_investing')
chromadb_path = r'chroma_db_ollama_predict_rts_21-00'
model_name = "bge-m3"
url_ai = "http://localhost:11434/api/embeddings"


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


def compare_vectors():
    # Получение документа с next_bar="None"
    none_results = collection.query(
        query_texts=[""],  # Пустой запрос, используем фильтр
        n_results=1,  # Только один документ
        where={"next_bar": "None"}
    )

    # Вывод даты из метаданных документа с next_bar="None"
    # print(none_results['metadatas'][0][0]['date'])

    if not none_results['ids'][0]:
        print("Документ с next_bar='None' не найден.")
        return

    none_id = none_results['ids'][0][0]
    none_embedding = collection.get(ids=[none_id], include=["embeddings"])["embeddings"][0]

    # Получение всех документов, кроме документа с next_bar="None"
    all_results = collection.query(
        query_texts=[""],
        n_results=1000,  # Достаточно большой лимит
        where={"next_bar": {"$ne": "None"}}  # Исключаем next_bar="None"
    )

    if not all_results['ids'][0]:
        print("Другие документы не найдены.")
        return

    # Получение векторов и метаданных для всех документов
    all_ids = all_results['ids'][0]
    all_embeddings = collection.get(ids=all_ids, include=["embeddings", "metadatas"])["embeddings"]
    all_metadatas = collection.get(ids=all_ids, include=["metadatas"])["metadatas"]

    # Вычисление косинусного сходства
    similarities = [
        (cosine_similarity(none_embedding, emb) * 100, meta, doc_id)
        for emb, meta, doc_id in zip(all_embeddings, all_metadatas, all_ids)
    ]

    # Сортировка по убыванию сходства
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Вывод метаданных и процента сходства для трех ближайших документов
    print(f"\nТри ближайших документа по сходству с документом next_bar='None' "
          f"на {none_results['metadatas'][0][0]['date']}:")
    for i, (similarity, metadata, doc_id) in enumerate(similarities[:1], 1):
        print(f"\nДокумент {i}:")
        print(f"Процент сходства: {similarity:.2f}%")
        print("Метаданные:")
        for key in sorted(metadata.keys()):
            print(f"  {key}: {metadata[key]}")


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Размер в МБ


def load_markdown_files(directory):
    documents = []
    for file_path in list(directory.glob("**/*.md")):
        # Чтение содержимого файла
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Разделение метаданных и текста
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                metadata_yaml = parts[1].strip()
                text_content = parts[2].strip()
                # Парсинг метаданных
                metadata = yaml.safe_load(metadata_yaml) or {}
                # Преобразование метаданных в строки
                metadata_str = {
                    "next_bar": str(metadata.get("next_bar", "unknown")),
                    "date_min": str(metadata.get("date_min", "unknown")),
                    "date_max": str(metadata.get("date_max", "unknown")),
                    "source": file_path.name,
                    "date": file_path.stem
                }
                # Создание объекта Document
                doc = Document(
                    page_content=text_content,
                    metadata=metadata_str
                )
                documents.append(doc)
            else:
                # Если нет метаданных, добавляем unknown
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
            # Если нет секции метаданных
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


# Удаление папки chroma_db, если она существует
if os.path.exists(chromadb_path):
    print(f"Размер папки {chromadb_path} до удаления: {get_folder_size(chromadb_path):.2f} МБ")
    shutil.rmtree(chromadb_path)
    print(f"Папка {chromadb_path} удалена.")

# Инициализация клиента ChromaDB
client = chromadb.PersistentClient(path=chromadb_path)

# Создание функции эмбеддингов для Ollama
ef = OllamaEmbeddingFunction(
    model_name=model_name,
    url=url_ai
)

# Создание коллекции
collection = client.create_collection(name="news_collection", embedding_function=ef)

# Загрузка Markdown-файлов
documents = load_markdown_files(md_path)

# Проверка на пустую папку
if not documents:
    print("Не найдено Markdown-файлов в указанной директории.")
    exit(1)
else:
    print(f"Загружено {len(documents)} Markdown-файлов из {md_path}")
    # print(f"Документы даты: {set(doc.metadata['date'] for doc in documents)}")
    print(f"Направление следующего бара: {set(doc.metadata['next_bar'] for doc in documents)}")
    # print(f"Минимальные даты: {set(doc.metadata['date_min'] for doc in documents)}")
    # print(f"Максимальные даты: {set(doc.metadata['date_max'] for doc in documents)}")

# Подготовка данных для ChromaDB
doc_texts = [doc.page_content for doc in documents]
doc_ids = [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in documents]
doc_metadatas = [doc.metadata for doc in documents]

# Добавление в коллекцию
try:
    collection.add(ids=doc_ids, documents=doc_texts, metadatas=doc_metadatas)
    print("Документы успешно добавлены.")
except Exception as e:
    print(f"Ошибка при добавлении документов: {e}")
    exit(1)

# Выполнение сравнения векторов
compare_vectors()