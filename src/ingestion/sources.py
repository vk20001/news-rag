"""
RSS Feed Sources Configuration
- Removed Hacker News: returns links/metadata, not article content
- Added Wired: good long-form tech content for chunking practice
"""

TECH_NEWS_FEEDS = [
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "category": "tech_general",
    },
    {
        "name": "Ars Technica",
        "url": "https://feeds.arstechnica.com/arstechnica/index",
        "category": "tech_deep",
    },
    {
        "name": "The Verge",
        "url": "https://www.theverge.com/rss/index.xml",
        "category": "tech_general",
    },
    {
        "name": "MIT Technology Review",
        "url": "https://www.technologyreview.com/feed/",
        "category": "tech_research",
    },
    {
        "name": "Wired",
        "url": "https://www.wired.com/feed/rss",
        "category": "tech_deep",
    },
]
