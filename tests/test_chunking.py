"""Tests for the chunking module."""
from src.chunking.chunker import fixed_size_chunk, recursive_chunk, chunk_article


def test_fixed_size_short_text():
    result = fixed_size_chunk("Short text", chunk_size=500)
    assert len(result) == 1
    assert result[0] == "Short text"

def test_fixed_size_splits():
    text = "A" * 1000
    result = fixed_size_chunk(text, chunk_size=500, overlap=50)
    assert len(result) > 1
    assert all(len(c) <= 500 for c in result)

def test_recursive_short_text():
    result = recursive_chunk("Short text", chunk_size=500)
    assert len(result) == 1

def test_recursive_respects_paragraphs():
    text = "First paragraph about topic A.\n\nSecond paragraph about topic B."
    result = recursive_chunk(text, chunk_size=50, overlap=0)
    assert len(result) == 2

def test_chunk_article_attaches_metadata():
    article = {
        "id": "test123",
        "title": "Test Article",
        "content": "This is the content of the test article. It has enough text to be processed.",
        "source": "TestSource",
        "url": "https://example.com/test",
        "published": "2026-01-01",
    }
    chunks = chunk_article(article)
    assert len(chunks) > 0
    assert chunks[0]["article_id"] == "test123"
    assert chunks[0]["source"] == "TestSource"
    assert chunks[0]["url"] == "https://example.com/test"

def test_chunk_article_empty_content():
    article = {"id": "empty", "content": "", "title": ""}
    assert chunk_article(article) == []
