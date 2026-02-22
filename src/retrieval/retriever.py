"""
Retrieval Module
================

CONCEPT: This is the Bridge Between Embedding and LLM
------------------------------------------------------
The embedding step stored 306 vectors in ChromaDB.
This module takes a user question, embeds it, and finds the
top-K most similar chunks.

These chunks become the "cheat sheet" we hand to the LLM.

CONCEPT: Top-K Retrieval
-------------------------
K = how many chunks to retrieve. Trade-off:
- Too few (K=1): might miss relevant info spread across chunks
- Too many (K=10): stuff irrelevant chunks into the prompt,
  confusing the LLM and wasting tokens

K=5 is a standard starting point. We'll make it configurable
so you can experiment later.

CONCEPT: Why Distance Matters
-----------------------------
ChromaDB returns chunks with a distance score.
Low distance = high similarity = more relevant.
We return this score so the hallucination gate can use it later —
if even the best chunk has high distance, the system probably
doesn't have good context to answer the question.
"""

import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:
    """
    Handles query embedding and chunk retrieval from ChromaDB.
    
    WHY A CLASS INSTEAD OF FUNCTIONS?
    The embedding model and ChromaDB client are expensive to create.
    Loading MiniLM takes ~2 seconds. Connecting to ChromaDB reads files
    from disk. You don't want to do this per query.
    
    A class loads them once in __init__ and reuses them across queries.
    In the Streamlit app, this object lives in session state —
    loaded once, used for every query the user makes.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "data/vectorstore",
        collection_name: str = "tech_news",
    ):
        print("Loading retriever...")
        self.model = SentenceTransformer(model_name)
        
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_collection(collection_name)
        
        print(f"Retriever ready. {self.collection.count()} vectors in store.")
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Given a user question, find the top-K most relevant chunks.
        
        Returns a list of dicts, each containing:
        - text: the chunk content
        - source: which news site
        - title: article title
        - url: link to original article
        - distance: how far from the query (lower = more relevant)
        
        This is what gets passed to the LLM as context.
        """
        # Embed the query using the same model that embedded the chunks
        # THIS IS CRITICAL: query and chunks must use the SAME model
        # otherwise the vectors live in different spaces and similarity
        # scores are meaningless
        query_vector = self.model.encode(query).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        
        # Package results into clean dicts
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "title": results["metadatas"][0][i].get("title", ""),
                "url": results["metadatas"][0][i].get("url", ""),
                "published": results["metadatas"][0][i].get("published", ""),
            })
        
        return retrieved
