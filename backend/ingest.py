import os
import sys

# Try to import with better error handling
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTrying to install missing dependencies...")
    print("Please run: pip install scipy numpy")
    sys.exit(1)

def ingest_data():
    # 1. Load Data
    loader = TextLoader("./data/benefits.txt")
    documents = loader.load()

    # 2. Split
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 3. Embed & Save (Locally)
    print("Generating local embeddings (HuggingFace)...")

    # Uses your CPU to create vectors. No API key needed.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory="./visa_db"
    )
    print(f"Success! Ingested {len(docs)} chunks locally.")

if __name__ == "__main__":
    ingest_data()
