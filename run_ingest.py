"""
CLI entry point. Calls the ingestion function.
Logic lives in src/ingestion/ — this just triggers it.
"""
from src.ingestion.rss_fetcher import ingest_all_feeds

if __name__ == "__main__":
    stats = ingest_all_feeds()
    if stats["new"] == 0 and stats["duplicate"] == 0:
        print("WARNING: No articles ingested. Check internet connection.")
        exit(1)
