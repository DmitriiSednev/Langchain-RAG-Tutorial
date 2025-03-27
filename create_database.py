import os
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from document_loaders import load_documents
from get_embedding_function import get_embedding_function

def check_env_vars():
    """Проверяет наличие необходимых переменных окружения"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Отсутствуют следующие переменные окружения:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nПожалуйста, добавьте их в файл .env")
        exit(1)

def main():
    # Загружаем переменные окружения
    load_dotenv()
    check_env_vars()
    
    # Пути к директориям
    data_path = "data"
    db_path = "chroma"
    
    # Очищаем старую базу данных если она существует
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Удалена старая база данных: {db_path}")
    
    # Загружаем документы
    print("Загрузка документов...")
    documents = load_documents(data_path)
    
    if not documents:
        print("Нет документов для обработки")
        return
    
    # Разбиваем документы на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    print(f"Разбито на {len(splits)} чанков")
    
    # Создаем эмбеддинги и сохраняем в базу
    print("Создание эмбеддингов и сохранение в базу...")
    embedding_function = get_embedding_function()
    db = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=db_path
    )
    db.persist()
    print(f"База данных успешно создана и сохранена в {db_path}")

if __name__ == "__main__":
    main()