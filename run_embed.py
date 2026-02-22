"""
CLI entry point for embedding.
Run AFTER run_chunk.py.

Usage:
    python run_embed.py
"""
from src.embedding.embedder import embed_and_store

if __name__ == "__main__":
    stats = embed_and_store()
