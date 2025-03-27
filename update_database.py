import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from document_loaders import load_documents
from get_embedding_function import get_embedding_function
import shutil


def update_database():
    """Обновляет базу данных, добавляя новые документы"""
    # Загружаем переменные окружения
    load_dotenv()

    # Проверяем наличие необходимых переменных
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Отсутствуют следующие переменные окружения:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nПожалуйста, добавьте их в файл .env")
        return

    # Получаем функцию эмбеддингов
    embedding_function = get_embedding_function()

    # Проверяем существование базы данных
    if not os.path.exists("chroma"):
        print("База данных не найдена. Создаем новую...")
        documents = load_documents()
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory="chroma",
        )
        print("База данных успешно создана!")
        return

    # Загружаем существующие документы
    print("Загружаем существующие документы...")
    existing_docs = load_documents()

    # Создаем временную базу данных
    print("Создаем временную базу данных...")
    temp_db = Chroma.from_documents(
        documents=existing_docs,
        embedding=embedding_function,
        persist_directory="chroma_temp",
    )

    # Удаляем старую базу данных
    print("Удаляем старую базу данных...")
    shutil.rmtree("chroma")

    # Перемещаем временную базу на место старой
    print("Обновляем базу данных...")
    shutil.move("chroma_temp", "chroma")

    print("База данных успешно обновлена!")


if __name__ == "__main__":
    update_database()
