# in this file, we are going to load and chunk up all our resources and chunk it up on our db
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

def load_documents(docs_path="docs"):
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader, 
        loader_kwargs={"encoding": 'utf-8'}
    )
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore


def main():
    docs_path = 'docs'
    persistent_directory = 'db/chroma_db'
    
    if os.path.exists(persistent_directory):
        embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        vectorstore= Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={'hnsw:space':'cosine'}
        )
        return vectorstore
    
    # 1. Load the files
    documents = load_documents(docs_path)
    # 2. Chunk the files
    chunks = split_documents(documents)
    # 3. Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks, persistent_directory)

if __name__ == "__main__":
    main()