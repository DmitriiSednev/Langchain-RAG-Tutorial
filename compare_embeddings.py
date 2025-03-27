import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from get_embedding_function import get_embedding_function
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import time
from typing import List, Dict
import json


def get_llm():
    """Возвращает LLM в зависимости от настроек"""
    llm_type = os.getenv("LLM_TYPE", "openai").lower()

    if llm_type == "openai":
        return ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
    elif llm_type == "ollama":
        return Ollama(model="llama2", temperature=0)
    else:
        raise ValueError(f"Неподдерживаемый тип LLM: {llm_type}")


def create_chain(embedding_type: str) -> RetrievalQA:
    """Создает цепочку с указанным типом эмбеддингов"""
    # Устанавливаем тип эмбеддингов
    os.environ["EMBEDDING_TYPE"] = embedding_type

    # Получаем функцию эмбеддингов
    embedding_function = get_embedding_function()

    # Инициализируем базу данных
    db = Chroma(persist_directory="chroma", embedding_function=embedding_function)

    # Создаем промпт
    prompt_template = """Используй следующие части контекста для ответа на вопрос. Если ты не знаешь ответа, просто скажи, что не знаешь. Не пытайся придумать ответ.

Контекст: {context}

Вопрос: {question}

Ответ:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Создаем цепочку
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
    )


def test_chain(chain: RetrievalQA, questions: List[str]) -> Dict:
    """Тестирует цепочку на заданных вопросах"""
    results = []
    total_time = 0

    for question in questions:
        start_time = time.time()
        result = chain.invoke({"query": question})
        end_time = time.time()

        results.append(
            {
                "question": question,
                "answer": result["result"],
                "time": end_time - start_time,
            }
        )
        total_time += end_time - start_time

    return {"results": results, "average_time": total_time / len(questions)}


def main():
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

    # Тестовые вопросы
    test_questions = [
        "сколько очков дается за самый длинный поезд в игре ticket_to_ride",
        "какие правила игры в монополию",
        "что происходит в начале истории Алисы в стране чудес",
    ]

    # Типы эмбеддингов для сравнения
    embedding_types = ["openai", "ollama"]

    # Сравниваем результаты
    comparison_results = {}

    for embedding_type in embedding_types:
        print(f"\nТестирование эмбеддингов типа: {embedding_type}")
        chain = create_chain(embedding_type)
        results = test_chain(chain, test_questions)
        comparison_results[embedding_type] = results

        # Выводим результаты
        print(f"\nСреднее время ответа: {results['average_time']:.2f} секунд")
        print("\nОтветы на вопросы:")
        for result in results["results"]:
            print(f"\nВопрос: {result['question']}")
            print(f"Ответ: {result['answer']}")
            print(f"Время: {result['time']:.2f} секунд")

    # Сохраняем результаты в файл
    with open("embedding_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    print("\nРезультаты сравнения сохранены в файл embedding_comparison.json")


if __name__ == "__main__":
    main()
