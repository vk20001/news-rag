"""
Routing Configuration
=====================
"""

DOMAIN_DESCRIPTION = "recent technology news including AI, software, hardware, tech companies, cybersecurity, and the tech industry"

FEW_SHOT_EXAMPLES = [
    # ANSWERABLE
    ("What did Apple announce this week?", "ANSWERABLE"),
    ("Any news about OpenAI?", "ANSWERABLE"),
    ("What is Microsoft doing in AI?", "ANSWERABLE"),
    ("Which cybersecurity threats are trending?", "ANSWERABLE"),
    ("What happened at MWC 2026?", "ANSWERABLE"),

    # OUT_OF_SCOPE
    ("What is the capital of France?", "OUT_OF_SCOPE"),
    ("What should I cook for dinner?", "OUT_OF_SCOPE"),
    ("What is OpenAI's stock price?", "OUT_OF_SCOPE"),
    ("Who won the football match yesterday?", "OUT_OF_SCOPE"),
    ("What is the weather like today?", "OUT_OF_SCOPE"),

    # AMBIGUOUS
    ("Tell me more", "AMBIGUOUS"),
    ("What happened?", "AMBIGUOUS"),
    ("And?", "AMBIGUOUS"),
    ("What about the other thing?", "AMBIGUOUS"),

    # SOCIAL
    ("ok got it thanks", "SOCIAL"),
    ("thanks for the update", "SOCIAL"),
    ("great thanks", "SOCIAL"),
    ("ok", "SOCIAL"),
    ("hello", "SOCIAL"),
    ("hi there", "SOCIAL"),
    ("hi", "SOCIAL"),
    ("hey", "SOCIAL"),
    ("that's interesting", "SOCIAL"),
    ("got it", "SOCIAL"),
    ("makes sense", "SOCIAL"),
    ("cool", "SOCIAL"),
    ("nice", "SOCIAL"),
    ("wow", "SOCIAL"),
    ("haha", "SOCIAL"),
    ("lol", "SOCIAL"),
    ("how's ur day today", "SOCIAL"),
    ("how are you", "SOCIAL"),
    ("how are you doing", "SOCIAL"),
    ("what's up", "SOCIAL"),
    ("whats up", "SOCIAL"),
    ("you good", "SOCIAL"),
]

COVERAGE_DISTANCE_THRESHOLD = 0.65
COVERAGE_PROBE_TOP_K = 3

# Pure greetings
GREETING_KEYWORDS = {"hi", "hey", "hello", "hie", "hii", "heya", "howdy", "sup", "yo", "hola"}

# Questions about the bot itself
BOT_QUESTION_KEYWORDS = {
    "how are you", "how r u", "how are u", "how's ur day",
    "how is your day", "you good", "what's up", "whats up",
    "how do you do", "how are you doing", "how r you"
}

GREETING_RESPONSES = [
    "Hey! 👋 What's going on in tech today — anything you want to dig into?",
    "Hi there! I'm across the latest tech news — what do you want to know?",
    "Hey! Ask me anything about recent tech — AI, gadgets, companies, all of it.",
]

BOT_QUESTION_RESPONSES = [
    "Ha, I'm just a news bot so no real days for me 😄 — but I'm ready! What tech news are you curious about?",
    "I'm good as long as there's tech news to talk about 😄 What do you want to know?",
    "Always doing great when there's interesting tech to discuss! What's on your mind?",
]

ACKNOWLEDGEMENT_RESPONSES = [
    "Glad that helped! Anything else you're curious about?",
    "Of course! What else do you want to know?",
    "Happy to help — what's next?",
    "Sure thing! Got more questions?",
]

OUT_OF_SCOPE_RESPONSES = [
    "That one's outside my lane — I only know tech news. Try asking about AI, startups, gadgets, or what's happening in the industry.",
    "I'm just a tech news nerd unfortunately 😅 — ask me about AI, software, hardware, or what's happening in the tech world.",
    "Not my area! I stick to tech news. What's going on in the tech world that you want to know about?",
]

AMBIGUOUS_RESPONSES = [
    "Can you be a bit more specific? Like — which company, product, or topic are you thinking of?",
    "I want to help but I need a bit more to go on — what exactly are you curious about?",
    "Give me a little more context and I'm on it! What specifically did you want to know?",
]

LOW_COVERAGE_RESPONSES = [
    "Hmm, that's a tech topic but I don't have much on it in my sources right now. Try asking about something that's been in the news recently.",
    "I don't have enough coverage on that one yet — my sources might not have picked it up. Anything else I can help with?",
]