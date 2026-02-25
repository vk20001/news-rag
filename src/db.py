# src/db.py
import os
import chromadb

def get_chroma_collection(
    persist_dir: str = "data/vectorstore",
    collection_name: str = "tech_news",
) -> chromadb.Collection:
    mode = os.getenv("CHROMA_MODE", "local")
    
    if mode == "http":
        host = os.getenv("CHROMA_HOST", "chromadb")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        print(f"ChromaDB mode: HTTP ({host}:{port})")
        client = chromadb.HttpClient(host=host, port=port)
    else:
        os.makedirs(persist_dir, exist_ok=True)
        print(f"ChromaDB mode: local ({persist_dir})")
        client = chromadb.PersistentClient(path=persist_dir)
    
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
