"""
Embedding + Vector Store Module
================================

CONCEPT: What Happens in This File
-----------------------------------
1. Load the embedding model (all-MiniLM-L6-v2)
2. Read your 373 chunks from data/processed/chunks.json
3. Pass each chunk's text through the model → get a 384-dim vector
4. Store text + vector + metadata in ChromaDB

After this, your data is SEARCHABLE BY MEANING.

CONCEPT: What is ChromaDB?
--------------------------
ChromaDB is a vector database. Regular databases (PostgreSQL, SQLite)
store rows and let you search by exact values: WHERE name = 'Meta'.

ChromaDB stores vectors and lets you search by SIMILARITY.
You give it a query vector, it returns the vectors closest to it.
"Closest" = highest cosine similarity = most similar meaning.

Think of it as: regular DB = dictionary lookup. Vector DB = "find me
things that mean something similar to this."

CONCEPT: Why We Embed Chunks, Not Full Articles
------------------------------------------------
We covered this — a 17,000-char article embedded as one vector produces
a blurred representation. Chunk-level embeddings are precise.

CONCEPT: Cosine Similarity (How "Closest" is Measured)
------------------------------------------------------
Two vectors are compared by the angle between them.
- Angle = 0 degrees → cosine = 1.0 → identical meaning
- Angle = 90 degrees → cosine = 0.0 → completely unrelated
- Angle = 180 degrees → cosine = -1.0 → opposite meaning

Your MiniLM model is trained so that semantically similar texts
produce vectors with high cosine similarity. That's the entire
foundation of semantic search.
"""

import json
import os
from sentence_transformers import SentenceTransformer
from src.db import get_chroma_collection


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load the sentence-transformers embedding model.
    
    First call downloads the model (~80MB). Subsequent calls use cache.
    Model runs on CPU — no GPU needed.
    
    This model takes any text and outputs a 384-dimensional vector.
    Max input: 256 tokens. Anything beyond is silently truncated.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model





# ──────────────────────────────────────────────
# Embedding + Storage
# ──────────────────────────────────────────────

def embed_and_store(
    chunks_path: str = "data/processed/chunks.json",
    persist_dir: str = "data/vectorstore",
    collection_name: str = "tech_news",
    batch_size: int = 32,
) -> dict:
    """
    Main function: load chunks, embed them, store in ChromaDB.
    
    WHY BATCH_SIZE = 32?
    Embedding one chunk at a time is slow (Python loop overhead).
    Embedding all 373 at once might spike RAM.
    Batches of 32 = good balance for 8GB RAM.
    
    DEDUPLICATION:
    ChromaDB uses chunk_id as the unique key. If you run this twice,
    it skips already-stored chunks. Same pattern as our JSON dedup.
    """
    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"\n{'='*50}")
    print(f"Embedding Pipeline")
    print(f"{'='*50}")
    print(f"Chunks to process: {len(chunks)}")
    
    # Load model
    model = load_embedding_model()
    
    # Connect to ChromaDB
    collection = get_chroma_collection(persist_dir, collection_name)
    
    # Check which chunks are already stored (dedup)
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    
    if not new_chunks:
        print("All chunks already embedded. Nothing to do.")
        return {"total": len(chunks), "new": 0, "skipped": len(chunks)}
    
    print(f"New chunks to embed: {len(new_chunks)} (skipping {len(existing_ids)} existing)")
    
    # Process in batches
    stats = {"total": len(chunks), "new": 0, "skipped": len(existing_ids), "errors": 0}
    
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(new_chunks) + batch_size - 1) // batch_size
        
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        try:
            # Extract texts for embedding
            texts = [c["text"] for c in batch]
            
            # Generate embeddings — this is where MiniLM does its work
            # Each text becomes a 384-dim vector
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            
            # Prepare data for ChromaDB
            ids = [c["chunk_id"] for c in batch]
            
            # Metadata: everything except the text and chunk_id
            # ChromaDB stores metadata alongside vectors for filtering
            metadatas = [
                {
                    "article_id": c["article_id"],
                    "source": c["source"],
                    "url": c["url"],
                    "title": c["title"],
                    "published": c["published"] or "",
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                }
                for c in batch
            ]
            
            # Add to ChromaDB: vectors + texts + metadata + unique IDs
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            
            stats["new"] += len(batch)
            
        except Exception as e:
            print(f"  ERROR in batch {batch_num}: {e}")
            stats["errors"] += len(batch)
    
    print(f"\nEmbedding Results:")
    print(f"  Total chunks:    {stats['total']}")
    print(f"  Newly embedded:  {stats['new']}")
    print(f"  Already existed: {stats['skipped']}")
    print(f"  Errors:          {stats['errors']}")
    print(f"  ChromaDB count:  {collection.count()}")
    print(f"  Vector store:    {persist_dir}/")
    print(f"{'='*50}\n")
    
    return stats
