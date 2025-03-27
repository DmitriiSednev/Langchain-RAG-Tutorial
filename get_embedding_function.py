from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


def get_embedding_function():
    """
    Возвращает функцию эмбеддингов в зависимости от настроек окружения.
    Поддерживаются:
    - OpenAI (по умолчанию)
    - Ollama
    - AWS Bedrock
    """
    embedding_type = os.getenv("EMBEDDING_TYPE", "openai").lower()

    if embedding_type == "ollama":
        return OllamaEmbeddings(
            model="nomic-embed-text",
            # base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    elif embedding_type == "bedrock":
        return BedrockEmbeddings(
            credentials_profile_name=os.getenv("AWS_PROFILE", "default"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    else:  # OpenAI по умолчанию
        return OpenAIEmbeddings()
