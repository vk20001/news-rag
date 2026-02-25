"""
CLI entry point for embedding.
Run AFTER run_chunk.py.

Usage:
    python run_embed.py
"""

from src.embedding.embedder import embed_and_store
from src.db import get_chroma_collection
from src.monitoring.drift_detector import detect_embedding_drift

if __name__ == "__main__":
    # Get existing embeddings BEFORE adding new ones
    collection = get_chroma_collection()
    existing = collection.get(include=["embeddings"])
    existing_embeddings = existing["embeddings"] if existing["embeddings"] is not None else []

    # Run embedding
    stats = embed_and_store()

    # Get new embeddings (the ones just added)
    if stats["new"] > 0:
        updated = collection.get(include=["embeddings"])
        all_embeddings = updated["embeddings"]
        new_embeddings = all_embeddings[len(existing_embeddings):]

        drift_result = detect_embedding_drift(existing_embeddings, new_embeddings)
        print(f"\nDrift Check: {drift_result['message']}")
        if drift_result["drift_detected"]:
            print("Recommendation: review new content — topic distribution has shifted.")
    else:
        print("\nDrift Check: skipped — no new chunks embedded.")
