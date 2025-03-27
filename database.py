import os
from dotenv import load_dotenv
from typing import List
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""


class ChromaDatabase:
    def __init__(self):
        # Загружаем переменные окружения
        load_dotenv()

        # Проверяем наличие необходимых переменных
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )

        self.embedding_function = get_embedding_function()
        self.db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=self.embedding_function
        )

    def _generate_chunk_id(
        self, chunk: Document, page_id: str, chunk_index: int
    ) -> str:
        """Генерирует уникальный идентификатор для чанка"""
        return f"{chunk.metadata.get('source', '')}:{page_id}:{chunk_index}"

    def add_documents(self, chunks: List[Document]) -> None:
        """Добавляет новые документы в базу данных или обновляет существующие"""
        # Получаем существующие ID документов
        existing_ids = set()
        if os.path.exists(CHROMA_PATH):
            existing_ids = set(self.db.get()["ids"]) if self.db.get()["ids"] else set()

        # Подготавливаем новые чанки с уникальными ID
        chunks_with_ids = []
        new_chunk_ids = []

        # Отслеживаем последний ID страницы для правильной нумерации чанков
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            # Если страница та же самая, увеличиваем индекс чанка
            current_page_id = (
                f"{chunk.metadata.get('source', '')}:{chunk.metadata.get('page', 0)}"
            )
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            last_page_id = current_page_id

            # Создаем уникальный ID для чанка
            chunk_id = self._generate_chunk_id(
                chunk, str(chunk.metadata.get("page", 0)), current_chunk_index
            )

            # Если чанк новый или требует обновления, добавляем его
            if chunk_id not in existing_ids:
                chunk.metadata["id"] = chunk_id
                chunk.metadata["chunk_index"] = current_chunk_index
                chunks_with_ids.append(chunk)
                new_chunk_ids.append(chunk_id)

        # Добавляем только новые документы
        if chunks_with_ids:
            self.db.add_documents(chunks_with_ids, ids=new_chunk_ids)
            print(f"Добавлено {len(chunks_with_ids)} новых документов")
        else:
            print("Новых документов не найдено")

    def query(self, query_text: str, k: int = 5) -> str:
        """Выполняет поиск по базе данных и возвращает ответ на основе контекста"""
        # Получаем k наиболее релевантных результатов
        results = self.db.similarity_search_with_score(query_text, k=k)

        # Форматируем контекст, добавляя источник документа
        context_texts = []
        print("\nНайденные релевантные фрагменты:")
        print("-" * 50)

        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)
            chunk_index = doc.metadata.get("chunk_index", 0)

            # Выводим информацию о фрагменте
            print(f"\nИсточник: {source}")
            print(f"Страница: {page}")
            print(f"Индекс чанка: {chunk_index}")
            print(f"Релевантность: {score:.4f}")
            print(f"Текст: {doc.page_content}\n")
            print("-" * 50)

            # Добавляем в контекст для модели
            context_texts.append(
                f"[Source: {source}, Page: {page}, Chunk: {chunk_index}, Score: {score:.4f}]\n{doc.page_content}"
            )

        context_text = "\n\n---\n\n".join(context_texts)

        # Создаем промпт
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Получаем ответ от модели
        model = ChatOpenAI()
        response = model.invoke(prompt)

        return response.content


def main():
    db = ChromaDatabase()

    while True:
        query = input("\nВведите ваш вопрос (или 'выход' для завершения): ")
        if query.lower() == "выход":
            break

        try:
            # Метод query уже выводит фрагменты и возвращает ответ
            response = db.query(query)
            print("\nОтвет:", response)
        except Exception as e:
            print(f"\nПроизошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()
