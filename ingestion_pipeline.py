# in this file, we are going to load and chunk up all our resources and chunk it up on our db
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

def main():
    print("Main Function")
    # 1. Load the files
    # 2. Chunk the files
    # 3. Embedding and Storing in Vector DB

if __name__ == "__main__":
    main()