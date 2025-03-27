import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from get_embedding_function import get_embedding_function
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama


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

    # Инициализируем базу данных
    embedding_function = get_embedding_function()
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
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
    )

    # Запускаем интерактивный режим
    print("Введите 'выход' для завершения")
    while True:
        question = input("\nВаш вопрос: ")
        if question.lower() == "выход":
            break

        result = chain.invoke({"query": question})
        print("\nОтвет:", result["result"])


if __name__ == "__main__":
    main()
