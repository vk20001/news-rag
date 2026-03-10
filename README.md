# Tech News RAG Pipeline

RAG pipeline for tech news with a hallucination detection quality gate. Ingests from RSS feeds, answers questions grounded in retrieved sources, and scores every answer for faithfulness before serving it.

---

## What's Actually Here

**Two-stage query router** - Groq LLM classifier runs first (~0.4s) to catch off-topic and social queries before wasting a full retrieval + generation cycle (~10s). If it passes, a ChromaDB coverage probe checks whether the KB actually has relevant content. On-topic but uncovered queries get a honest "I don't have that" instead of a hallucinated answer.

**NLI hallucination gate** - cross-encoder scores the generated answer against retrieved chunks for faithfulness. Answers below threshold are flagged. Uses `cross-encoder/nli-deberta-v3-small` running locally - deterministic, no API cost, no quota dependency.

**Conversational memory with query rewriting** - follow-ups like "tell me more about their funding" are rewritten into standalone queries using conversation history before hitting retrieval. Without this, context-dependent queries retrieve nothing useful.

**Multi-provider LLM fallback** - Gemini 2.5 Flash primary, Groq Llama 3.3 70B fallback. Provider-agnostic via OpenAI-compatible SDK - switching providers is two lines.

---

## Demo

### Greeting + Real Question
![Greeting and Anthropic news query](docs/screenshots/greeting_and_query.png)

### Out-of-Scope Rejection + Hallucination Gate
![Out of scope and hallucination gate](docs/screenshots/out_of_scope_and_gate.png)

### Conversational Follow-up
![Conversational memory and knowledge gap](docs/screenshots/conversational_and_gap.png)

### Monitoring Dashboard
![Pipeline monitoring dashboard](docs/screenshots/dashboard.png)

---

## Architecture

```
[RSS Feeds: TechCrunch, Ars Technica, The Verge, MIT Tech Review,
           VentureBeat, Engadget, ZDNet, The Next Web]
    ↓
[Ingestion: feedparser + HTML stripping + URL-hash deduplication]
    ↓
[Recursive Chunking: 500 chars, 50 overlap]
    ↓
[Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-dim)]
    ↓
[ChromaDB — HTTP container, single source of truth]
    ↓
[User Query]
    ↓
┌─────────────────────────────────────────┐
│         TWO-STAGE QUERY ROUTER          │
│                                         │
│  Stage 1: Groq classifier (~0.4s)       │
│    SOCIAL       → direct response       │
│    OUT_OF_SCOPE → rejection             │
│    AMBIGUOUS    → clarification         │
│    ANSWERABLE   → Stage 2               │
│                                         │
│  Stage 2: ChromaDB coverage probe       │
│    distance > 0.65 → LOW_COVERAGE       │
│    distance ≤ 0.65 → full pipeline      │
└─────────────────────────────────────────┘
    ↓
[Query Rewriter — resolves follow-up references]
    ↓
[Top-5 Semantic Retrieval]
    ↓
[Versioned Prompt (YAML) + Chunks + Conversation History]
    ↓
[Gemini 2.5 Flash / Groq Llama 3.3 70B fallback]
    ↓
[NLI Hallucination Gate]
    ├── score ≥ 0.65 → serve answer
    └── score < 0.65 → flag low-confidence
    ↓
[SQLite metrics → Streamlit dashboard]
```

---

## Routing Behaviour

| Query | Stage 1 | Stage 2 | Result |
|-------|---------|---------|--------|
| "hey" | SOCIAL | - | Casual response, 0.4s |
| "how are u" | SOCIAL | - | Bot response, 0.4s |
| "what's the weather" | OUT_OF_SCOPE | - | Rejection, 0.4s |
| "tell me about OpenAI" | ANSWERABLE | PASS | Full answer, ~10s |
| "OpenAI stock price" | OUT_OF_SCOPE | - | Rejection, 0.4s |
| "Anthropic Opus 4.6 BrowseComp" | ANSWERABLE | FAIL | Low coverage response |
| "tell me more" | AMBIGUOUS | - | Clarification request |

---

## Stack

| Layer | Tool |
|-------|------|
| Data | RSS feeds (8 sources) |
| Chunking | Recursive text splitter |
| Embedding | all-MiniLM-L6-v2 |
| Vector store | ChromaDB (HTTP) |
| Router Stage 1 | Groq LLM classifier |
| Router Stage 2 | ChromaDB coverage probe |
| Query rewriter | Groq |
| LLM | Gemini 2.5 Flash + Groq Llama 3.3 70B |
| Hallucination gate | cross-encoder/nli-deberta-v3-small |
| Prompt management | Versioned YAML (v1/v2/v3) |
| Monitoring | SQLite + Streamlit |
| CI/CD | GitHub Actions + pytest |
| Containers | Docker + Compose |

---

## Setup

```bash
git clone https://github.com/vk20001/news-rag.git
cd news-rag

cp .env.example .env
# Add GEMINI_API_KEY (ai.google.dev) and GROQ_API_KEY (console.groq.com)
# Both free, no credit card

docker compose up -d
docker compose --profile ingest run ingest
# Open http://localhost:8501
```

```bash
pytest tests/ -v                          # 20 tests
python run_query.py "Any news on OpenAI"  # CLI query
```

---

## What I Learned

**NLI on real-world text behaves differently than benchmarks suggest** - cross-encoders trained on clean NLI datasets score long, messy news chunks inconsistently. Label mappings and thresholds needed empirical calibration, not the defaults from the model card.

**ChromaDB sync was the hardest infrastructure problem** - local ingestion writing to `data/vectorstore/` while the Docker app read from a separate volume meant two out-of-sync stores silently. Fixed by routing both through the same ChromaDB HTTP container via `CHROMA_MODE=http`.

**Query rewriting is non-negotiable for multi-turn RAG** - vague follow-ups produce bad retrieval. Rewriting "tell me more about their funding" to "What is OpenAI's latest funding situation?" before hitting ChromaDB is what makes conversation actually work.

**Two-stage routing changes the economics** - rejecting off-topic queries at the classifier level costs ~0.4s and near-zero API tokens. Letting them fall through to full generation wastes ~10s and burns quota on answers that should never have been attempted.

