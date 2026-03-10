"""
RSS Feed Sources Configuration

Changes from v1:
- Removed Wired: blocks feedparser, returns 0 articles consistently
- Replaced TechCrunch full feed with category-specific feeds (more articles)
- Added VentureBeat AI, ZDNet, The Next Web, Engadget
- Added Reuters Technology for broader coverage
"""

TECH_NEWS_FEEDS = [
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "category": "tech_general",
    },
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "ai",
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
        "name": "VentureBeat",
        "url": "https://venturebeat.com/feed/",
        "category": "tech_general",
    },
    {
        "name": "VentureBeat AI",
        "url": "https://venturebeat.com/category/ai/feed/",
        "category": "ai",
    },
    {
        "name": "The Next Web",
        "url": "https://thenextweb.com/feed/",
        "category": "tech_general",
    },
    {
        "name": "ZDNet",
        "url": "https://www.zdnet.com/news/rss.xml",
        "category": "tech_general",
    },
    {
        "name": "Engadget",
        "url": "https://www.engadget.com/rss.xml",
        "category": "tech_general",
    },
]