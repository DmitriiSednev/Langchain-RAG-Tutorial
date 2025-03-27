from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from langchain.schema import Document
import os
import glob
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from database import ChromaDatabase

# Маппинг расширений файлов к соответствующим загрузчикам
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
}


def load_documents(data_path: str) -> List[Document]:
    """Загружает документы из указанной директории"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Загружаем все файлы из директории
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if not os.path.isfile(file_path):
            continue

        # Определяем тип файла и загружаем соответствующим лоадером
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif filename.endswith((".doc", ".docx")):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            print(f"Пропускаем файл {filename}: неподдерживаемый формат")
            continue

        try:
            # Загружаем документ
            docs = loader.load()

            # Разбиваем на чанки
            chunks = text_splitter.split_documents(docs)

            # Отладочная информация
            print(f"\nАнализ файла {filename}:")
            print(f"Количество страниц в документе: {len(docs)}")
            print(f"Количество чанков: {len(chunks)}")
            print("\nПример чанка (189):")
            if len(chunks) > 189:
                print(f"Страница: {chunks[189].metadata.get('page', 'не указана')}")
                print(
                    f"Индекс чанка: {chunks[189].metadata.get('chunk_index', 'не указан')}"
                )
                print(f"Текст: {chunks[189].page_content[:200]}...")

            # Добавляем информацию об источнике и странице
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = filename
                # Сохраняем номер страницы из метаданных документа
                if "page" in chunk.metadata:
                    # Добавляем +1 к номеру страницы, так как нумерация начинается с 0
                    chunk.metadata["page"] = chunk.metadata["page"] + 1
                else:
                    # Если номер страницы не указан, пробуем определить по содержимому
                    if "page" in chunk.page_content.lower():
                        try:
                            page_num = int(
                                chunk.page_content.split("page")[1].split()[0]
                            )
                            chunk.metadata["page"] = page_num
                        except:
                            chunk.metadata["page"] = 1
                    else:
                        chunk.metadata["page"] = 1
                # Добавляем индекс чанка
                chunk.metadata["chunk_index"] = i
                # Добавляем общий индекс для отслеживания порядка
                chunk.metadata["index"] = i

            documents.extend(chunks)
            print(f"Загружен файл: {filename}")

        except Exception as e:
            print(f"Ошибка при загрузке файла {filename}: {str(e)}")

    return documents


def create_database(data_path: str = "data") -> None:
    """Создает базу данных из документов в указанной директории"""
    print("Загрузка документов...")
    documents = load_documents(data_path)

    print("Создание базы данных...")
    db = ChromaDatabase()
    db.add_documents(documents)

    print("База данных успешно создана!")


if __name__ == "__main__":
    create_database()
