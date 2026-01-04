"""
Simplified ingestion script that avoids problematic imports
Uses direct file reading instead of TextLoader
"""

import os
from pathlib import Path

# Direct imports to avoid langchain_text_splitters dependency chain
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
except ImportError as e:
    print(f"Error importing: {e}")
    print("Please install: pip install langchain-huggingface langchain-chroma")
    exit(1)

def ingest_data():
    # 1. Load Data directly (avoid TextLoader)
    data_path = Path("./data/benefits.txt")
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 2. Split manually (avoid CharacterTextSplitter)
    # Split by double newlines (paragraphs)
    chunks = content.split('\n\n')
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Create Document objects
    documents = [
        Document(page_content=chunk, metadata={"source": "benefits.txt", "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]
    
    print(f"Loaded {len(documents)} chunks from benefits.txt")
    
    # 3. Embed & Save (Locally)
    print("Generating local embeddings (HuggingFace)...")
    print("This may take a minute on first run (downloading model)...")
    
    try:
        # Uses your CPU to create vectors. No API key needed.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Remove existing database if it exists
        if os.path.exists("./visa_db"):
            import shutil
            print("Removing existing database...")
            shutil.rmtree("./visa_db")
        
        db = Chroma.from_documents(
            documents, 
            embeddings, 
            persist_directory="./visa_db"
        )
        print(f"Success! Ingested {len(documents)} chunks locally.")
        print(f"Database saved to: ./visa_db")
        
    except Exception as e:
        print(f"Error during embedding: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure scipy is installed: pip install scipy")
        print("2. Try: pip install --upgrade scipy numpy")
        print("3. If still failing, try: ./fix_scipy.sh")
        raise

if __name__ == "__main__":
    ingest_data()

