"""
RSS Feed Fetcher + Deduplication
================================

1. DATA INGESTION LAYER (Bronze/Raw)
   Fetch from source → Parse → Deduplicate → Store raw
   We store raw JSON first (not vectors) because if you change chunking
   strategy later, you re-chunk from raw — don't re-fetch.

2. DEDUPLICATION VIA URL HASHING
   RSS feeds overlap across fetches. We hash each article's URL with SHA-256
   and use the hash as the filename. If file exists = already ingested = skip.
   O(1) dedup check, no database needed.

3. HTML STRIPPING
   RSS content comes with HTML tags (<p>, <a>, <img>).
   If you embed HTML into vectors, you pollute the semantic space with
   markup that has zero informational value. Strip it before storing.
"""

import feedparser
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from src.ingestion.sources import TECH_NEWS_FEEDS


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text. Regex handles 95% of RSS content HTML."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def hash_url(url: str) -> str:
    """
    SHA-256 hash of URL, truncated to 16 chars.
    Deterministic: same URL always = same hash.
    This hash becomes the filename: data/raw/{hash}.json
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def fetch_single_feed(feed_url: str, feed_name: str, feed_category: str) -> list[dict]:
    """
    Fetch and parse one RSS feed.

    What feedparser does:
    RSS is XML. Each feed has <item> entries with <title>, <link>,
    <description>, <content:encoded>, <pubDate>. feedparser parses
    the XML and normalizes across RSS 2.0 / Atom / RDF formats.
    """
    print(f"  Fetching: {feed_name} ({feed_url})")

    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"  ERROR fetching {feed_name}: {e}")
        return []

    if feed.bozo and not feed.entries:
        print(f"  WARNING: {feed_name} feed has parsing issues and no entries")
        return []

    articles = []

    for entry in feed.entries:
        link = entry.get("link", "")
        if not link:
            continue

        # Prefer full content, fall back to summary
        content = ""
        if hasattr(entry, "content") and entry.content:
            content = entry.content[0].get("value", "")
        if not content:
            content = entry.get("summary", "") or entry.get("description", "")

        content = strip_html_tags(content)
        title = strip_html_tags(entry.get("title", ""))

        if len(content) < 200:
            continue

        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
            except (ValueError, TypeError):
                published = None

        article = {
            "id": hash_url(link),
            "title": title,
            "content": content,
            "url": link,
            "source": feed_name,
            "category": feed_category,
            "published": published,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        articles.append(article)

    print(f"  Found {len(articles)} articles from {feed_name}")
    return articles


def save_articles(articles: list[dict], raw_dir: str = "data/raw") -> dict:
    """
    Save articles as individual JSON files with dedup.
    
    Why individual files, not one big JSON?
    - Dedup = os.path.exists() — O(1), no file parsing
    - Each article independently addressable
    - Delete old articles = delete files, no JSON surgery
    """
    os.makedirs(raw_dir, exist_ok=True)
    stats = {"new": 0, "duplicate": 0, "error": 0}

    for article in articles:
        filepath = os.path.join(raw_dir, f"{article['id']}.json")

        if os.path.exists(filepath):
            stats["duplicate"] += 1
            continue

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article, f, ensure_ascii=False, indent=2)
            stats["new"] += 1
        except Exception as e:
            print(f"  ERROR saving article {article['id']}: {e}")
            stats["error"] += 1

    return stats


def ingest_all_feeds(
    feeds: Optional[list[dict]] = None,
    raw_dir: str = "data/raw",
) -> dict:
    """
    Main entry point: fetch all feeds, deduplicate, store raw JSON.
    
    This function is decoupled from orchestration — your CLI calls it,
    later your Airflow DAG calls the same function. Logic doesn't know
    or care who triggered it.
    """
    if feeds is None:
        feeds = TECH_NEWS_FEEDS

    print(f"\n{'='*50}")
    print(f"RSS Ingestion — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}")
    print(f"Feeds to process: {len(feeds)}")

    all_articles = []
    for feed in feeds:
        articles = fetch_single_feed(
            feed_url=feed["url"],
            feed_name=feed["name"],
            feed_category=feed["category"],
        )
        all_articles.extend(articles)

    print(f"\nTotal articles fetched: {len(all_articles)}")

    stats = save_articles(all_articles, raw_dir=raw_dir)

    print(f"\nIngestion Results:")
    print(f"  New articles saved:  {stats['new']}")
    print(f"  Duplicates skipped:  {stats['duplicate']}")
    print(f"  Errors:              {stats['error']}")
    print(f"  Raw storage:         {raw_dir}/")
    print(f"{'='*50}\n")

    return stats
