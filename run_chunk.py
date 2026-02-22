"""
CLI entry point for chunking.
Run AFTER run_ingest.py.

Usage:
    python run_chunk.py
"""
from src.chunking.chunker import chunk_all_articles

if __name__ == "__main__":
    stats = chunk_all_articles(
        strategy="recursive",
        chunk_size=500,
        overlap=50,
    )
