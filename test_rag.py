import pytest
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from get_embedding_function import get_embedding_function
from langchain_openai import ChatOpenAI


@pytest.fixture
def db():
    """Фикстура для создания тестовой базы данных"""
    embedding_function = get_embedding_function()
    # Создаем тестовые документы
    texts = [
        "Это тестовый документ для проверки работы системы.",
        "Второй тестовый документ с дополнительной информацией.",
        "Третий документ для проверки поиска.",
    ]
    # Создаем новую базу и добавляем документы
    db = Chroma(persist_directory="test_chroma", embedding_function=embedding_function)
    db.add_texts(texts)
    return db


@pytest.fixture
def chain(db):
    """Фикстура для создания тестовой цепочки"""
    prompt_template = """Используй следующие части контекста для ответа на вопрос. Если ты не знаешь ответа, просто скажи, что не знаешь. Не пытайся придумать ответ.

Контекст: {context}

Вопрос: {question}

Ответ:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
    )


def test_chain_creation(chain):
    """Тест создания цепочки"""
    assert chain is not None
    assert isinstance(chain, RetrievalQA)


def test_query_response(chain):
    """Тест получения ответа на запрос"""
    result = chain.invoke({"query": "Что содержится в тестовых документах?"})
    assert result is not None
    assert "result" in result
    assert isinstance(result["result"], str)
    assert len(result["result"]) > 0


def test_empty_query(chain):
    """Тест обработки пустого запроса"""
    result = chain.invoke({"query": ""})
    assert result is not None
    assert "result" in result
    assert isinstance(result["result"], str)
