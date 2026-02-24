"""Tests for the ingestion module."""
from src.ingestion.rss_fetcher import strip_html_tags, hash_url


def test_strip_html_tags_basic():
    assert strip_html_tags("<p>Hello</p>") == "Hello"

def test_strip_html_tags_nested():
    assert strip_html_tags("<div><p>Hello <b>world</b></p></div>") == "Hello world"

def test_strip_html_tags_empty():
    assert strip_html_tags("") == ""

def test_strip_html_tags_no_html():
    assert strip_html_tags("Just plain text") == "Just plain text"

def test_hash_url_deterministic():
    url = "https://example.com/article"
    assert hash_url(url) == hash_url(url)

def test_hash_url_different_urls():
    assert hash_url("https://a.com") != hash_url("https://b.com")

def test_hash_url_length():
    assert len(hash_url("https://example.com")) == 16
