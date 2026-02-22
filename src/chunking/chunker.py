"""
Text Chunking Module
====================

CONCEPT: Why Chunking Matters for RAG
--------------------------------------
Your embedding model (all-MiniLM-L6-v2) converts text to a 384-dim vector.
The problem: it has a MAX INPUT of 256 tokens (~200 words).

If you feed it a 17,000-char article:
- It silently truncates to 256 tokens
- The vector only represents the FIRST ~200 words
- Everything after that is invisible to search

So you MUST split articles into chunks that fit within the model's limit.
But HOW you split determines retrieval quality.

CONCEPT: Fixed-Size vs Recursive Chunking
------------------------------------------
FIXED-SIZE:
  Split every N characters. Simple. But stupid.
  "The CEO of Apple, Tim Cook, announced" might get split into:
  Chunk 1: "The CEO of Apple, Tim"
  Chunk 2: "Cook, announced the new..."
  Now "Tim Cook" is split across chunks. A search for "Tim Cook" 
  won't match either chunk well.

RECURSIVE (what we use):
  Try to split at paragraph boundaries (\n\n) first.
  If a chunk is still too big, split at sentence boundaries (. ! ?)
  If still too big, split at word boundaries (spaces).
  This RESPECTS the natural structure of text.

CONCEPT: Overlap
----------------
Even with smart splitting, context at chunk boundaries gets lost.
Solution: chunks overlap by N characters.

Example with chunk_size=100, overlap=20:
  Chunk 1: chars 0-100
  Chunk 2: chars 80-180  (starts 20 chars before chunk 1 ended)
  Chunk 3: chars 160-260

The repeated 20 chars at each boundary act as "glue" — 
if a key sentence falls at a boundary, it appears in both chunks,
so at least one chunk will match the query.
"""

import json
import os
from typing import Optional


def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Simplest strategy: split every N characters with overlap.
    
    We implement this so you can COMPARE retrieval quality against
    recursive chunking later. Spoiler: this will perform worse.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk.strip())
        
        # Move forward by (chunk_size - overlap)
        start += chunk_size - overlap
    
    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Smart strategy: split at natural text boundaries.
    
    How it works:
    1. Try splitting on double newline (paragraph breaks)
    2. If any piece is still > chunk_size, split on single newline
    3. If still too big, split on ". " (sentence boundaries)
    4. If still too big, split on " " (word boundaries)
    5. Last resort: hard cut at chunk_size (same as fixed)
    
    This is the same logic LangChain's RecursiveCharacterTextSplitter uses.
    We're implementing it from scratch so you understand what's inside the box.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    # Find the best separator that exists in the text
    chosen_sep = None
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break
    
    # No separator found — hard cut (last resort)
    if chosen_sep is None:
        return fixed_size_chunk(text, chunk_size, overlap)
    
    # Split on the chosen separator
    pieces = text.split(chosen_sep)
    
    # Now merge pieces into chunks that respect chunk_size
    chunks = []
    current_chunk = ""
    
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        
        # If adding this piece would exceed chunk_size
        if current_chunk and len(current_chunk) + len(chosen_sep) + len(piece) > chunk_size:
            # Save current chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from end of previous
            if overlap > 0 and current_chunk:
                # Take last 'overlap' chars from previous chunk as context
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + chosen_sep + piece
            else:
                current_chunk = piece
        else:
            # Add piece to current chunk
            if current_chunk:
                current_chunk += chosen_sep + piece
            else:
                current_chunk = piece
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Recursively handle any chunks that are still too big
    final_chunks = []
    remaining_seps = separators[separators.index(chosen_sep) + 1:]
    
    for chunk in chunks:
        if len(chunk) > chunk_size and remaining_seps:
            # Try next separator level
            sub_chunks = recursive_chunk(chunk, chunk_size, overlap, remaining_seps)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def chunk_article(article: dict, strategy: str = "recursive", chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Chunk a single article and attach metadata to each chunk.
    
    WHY METADATA ON EVERY CHUNK?
    When you retrieve a chunk later, you need to know:
    - Which article it came from (for citation/source link)
    - Which source published it (for trust ranking)
    - Where in the article it appeared (chunk index)
    
    Without metadata, your retrieved chunk is an orphan —
    you can't cite it, can't link to the original, can't tell
    the user where the information came from.
    """
    content = article.get("content", "")
    
    if not content.strip():
        return []
    
    # Prepend title to content — the title often contains key info
    # that's not repeated in the body. "Apple Releases iPhone 17"
    # as a title + body about features = the chunk needs both.
    title = article.get("title", "")
    if title:
        full_text = f"{title}\n\n{content}"
    else:
        full_text = content
    
    # Choose strategy
    if strategy == "fixed":
        raw_chunks = fixed_size_chunk(full_text, chunk_size, overlap)
    else:
        raw_chunks = recursive_chunk(full_text, chunk_size, overlap)
    
    # Attach metadata to each chunk
    chunked = []
    for i, chunk_text in enumerate(raw_chunks):
        chunked.append({
            "chunk_id": f"{article['id']}_chunk_{i}",
            "article_id": article["id"],
            "text": chunk_text,
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "title": article.get("title", ""),
            "published": article.get("published", ""),
            "chunk_index": i,
            "total_chunks": len(raw_chunks),  # filled after loop
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
        })
    
    # Update total_chunks (now we know the final count)
    for chunk in chunked:
        chunk["total_chunks"] = len(chunked)
    
    return chunked


def chunk_all_articles(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
) -> dict:
    """
    Read all raw articles, chunk them, save processed chunks.
    
    Saves one JSON file per article containing all its chunks.
    Also saves a combined chunks.json for easy loading later.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    
    print(f"\n{'='*50}")
    print(f"Chunking — strategy: {strategy}, size: {chunk_size}, overlap: {overlap}")
    print(f"{'='*50}")
    print(f"Articles to process: {len(files)}")
    
    all_chunks = []
    stats = {"articles": 0, "total_chunks": 0, "empty_articles": 0}
    
    for filename in files:
        filepath = os.path.join(raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)
        
        chunks = chunk_article(article, strategy, chunk_size, overlap)
        
        if not chunks:
            stats["empty_articles"] += 1
            continue
        
        chunks = [c for c in chunks if len(c["text"]) >= 50]
        all_chunks.extend(chunks)
        stats["articles"] += 1
        stats["total_chunks"] += len(chunks)
    
    # Save combined file for easy loading in embedding step
    combined_path = os.path.join(output_dir, "chunks.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    stats["avg_chunks_per_article"] = round(stats["total_chunks"] / max(stats["articles"], 1), 1)
    
    # Calculate chunk length stats
    chunk_lengths = [len(c["text"]) for c in all_chunks]
    
    print(f"\nChunking Results:")
    print(f"  Articles processed: {stats['articles']}")
    print(f"  Empty articles:     {stats['empty_articles']}")
    print(f"  Total chunks:       {stats['total_chunks']}")
    print(f"  Avg chunks/article: {stats['avg_chunks_per_article']}")
    print(f"  Chunk length — min: {min(chunk_lengths)} chars")
    print(f"  Chunk length — max: {max(chunk_lengths)} chars")
    print(f"  Chunk length — avg: {sum(chunk_lengths) // len(chunk_lengths)} chars")
    print(f"  Output: {combined_path}")
    print(f"{'='*50}\n")
    
    return stats
