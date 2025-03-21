from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books/alice_in_wonderland.md"

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()

def load_documents():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    return [Document(page_content=text)]

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    main()
