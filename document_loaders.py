from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document
import os
import glob

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

def load_documents(data_path: str) -> list[Document]:
    """
    Загружает документы из указанной директории.
    Поддерживает рекурсивный поиск по поддиректориям.
    
    Args:
        data_path: Путь к директории с документами
        
    Returns:
        Список загруженных документов
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Создана директория {data_path}")
        return []
    
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(glob.glob(os.path.join(data_path, f"**/*{ext}"), recursive=True))
    
    if not all_files:
        print(f"В директории {data_path} не найдено поддерживаемых файлов.")
        print("Поддерживаемые форматы:", ", ".join(LOADER_MAPPING.keys()))
        return []
    
    documents = []
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in LOADER_MAPPING:
            loader_class = LOADER_MAPPING[ext]
            try:
                loader = loader_class(file_path)
                docs = loader.load()
                # Добавляем информацию об источнике
                for doc in docs:
                    doc.metadata["source"] = file_path
                documents.extend(docs)
                print(f"Загружен файл: {file_path}")
            except Exception as e:
                print(f"Ошибка загрузки {file_path}: {str(e)}")
    
    return documents