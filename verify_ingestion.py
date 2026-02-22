"""Run AFTER run_ingest.py to inspect what you got."""
import json
import os
import re

RAW_DIR = "data/raw"

def verify():
    if not os.path.exists(RAW_DIR):
        print("ERROR: data/raw/ doesn't exist. Run 'python run_ingest.py' first.")
        return

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]
    print(f"\nTotal articles in data/raw/: {len(files)}")

    if not files:
        print("No articles found.")
        return

    lengths = []
    sources = {}

    for filename in files:
        with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
            article = json.load(f)
        lengths.append(len(article.get("content", "")))
        source = article.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    print(f"\nArticles by source:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    print(f"\nContent length stats:")
    print(f"  Min:     {min(lengths)} chars")
    print(f"  Max:     {max(lengths)} chars")
    print(f"  Average: {sum(lengths) // len(lengths)} chars")

    # Show one sample
    with open(os.path.join(RAW_DIR, files[0]), "r", encoding="utf-8") as f:
        sample = json.load(f)

    print(f"\n{'='*50}")
    print(f"SAMPLE ARTICLE")
    print(f"{'='*50}")
    print(f"Title:     {sample['title']}")
    print(f"Source:    {sample['source']}")
    print(f"URL:       {sample['url']}")
    print(f"Published: {sample['published']}")
    print(f"Content:   {sample['content'][:300]}...")
    print(f"Length:    {len(sample['content'])} chars")

    # HTML contamination check
    html_pattern = re.compile(r"<[^>]+>")
    contaminated = sum(
        1 for fn in files[:20]
        if html_pattern.search(
            json.load(open(os.path.join(RAW_DIR, fn), "r", encoding="utf-8")).get("content", "")
        )
    )

    if contaminated:
        print(f"\nWARNING: {contaminated} articles still have HTML tags.")
    else:
        print(f"\nHTML check: Clean")

if __name__ == "__main__":
    verify()
