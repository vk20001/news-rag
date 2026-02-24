# Tech News RAG Pipeline with Hallucination Detection

A self-evaluating Retrieval-Augmented Generation pipeline that ingests tech news, answers questions grounded in real sources, and automatically detects hallucinated answers using NLI-based faithfulness scoring.

## What Makes This Different

Most RAG tutorials stop at "retrieve chunks → generate answer." This pipeline adds a **hallucination quality gate** — every answer is scored for faithfulness before being served. Low-confidence answers are flagged, not blindly delivered.

- **Hallucination detection** using NLI cross-encoder models (runs locally on CPU)
- **Measured prompt engineering** — v1 vs v2 A/B comparison with faithfulness metrics
- **Multi-provider LLM fallback** — Gemini primary, Groq backup, provider-agnostic SDK
- **Full monitoring** — every query logged to SQLite with faithfulness scores, latency, provider info
- **Streamlit UI** with query interface and monitoring dashboard
- **CI/CD** with automated tests in GitHub Actions

## Architecture
```
[RSS Feeds: TechCrunch, Ars Technica, The Verge, MIT Tech Review, Wired]
    ↓
[Ingestion: feedparser + HTML stripping + URL-hash deduplication]
    ↓
[Recursive Chunking: 500 chars, 50 overlap, min 50 char filter]
    ↓
[Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU)]
    ↓
[ChromaDB Vector Store (in-process, no server)]
    ↓
[Query → Top-5 Semantic Retrieval]
    ↓
[Prompt Template (versioned YAML)] + [Retrieved Chunks]
    ↓
[Google Gemini API (primary) / Groq API (fallback)]
    ↓
[Hallucination Quality Gate: NLI cross-encoder faithfulness scoring]
    ├── Score ≥ 0.5 → Serve answer with confidence score
    └── Score < 0.5 → Flag as low-confidence
    ↓
[Metrics logged to SQLite → Streamlit monitoring dashboard]
```

## Tech Stack

| Layer | Tool | Why This Choice |
|-------|------|-----------------|
| Data Source | RSS feeds (5 sources) | Real, updating data — not static CSV |
| Chunking | Recursive text splitter | Respects sentence boundaries, handles overlap |
| Embedding | all-MiniLM-L6-v2 | 80MB, CPU-friendly, 384-dim vectors |
| Vector Store | ChromaDB (in-process) | No server overhead, Python-native |
| LLM | Gemini 2.5 Flash + Groq Llama 3.3 70B | Both free, multi-provider fallback |
| Hallucination Gate | cross-encoder/nli-deberta-v3-small | 200MB, CPU, deterministic scoring |
| Monitoring | SQLite + Streamlit | Zero background memory cost |
| CI/CD | GitHub Actions + pytest | Automated test suite on every push |

## Key Design Decision: Lightweight-First

Entire pipeline runs on an **8GB RAM laptop with no GPU**. This proves that good GenAI engineering is about architecture decisions, not infrastructure budget.

- Embedding model: ~80MB on CPU
- NLI model: ~200MB on CPU
- LLM inference: remote API (zero local RAM)
- Vector store: in-process (no separate server)
- Monitoring: SQLite file (no database server)

## Prompt Engineering Experiment

Compared two prompt strategies using faithfulness scores across 5 test questions:

| Prompt | Strategy | Avg Faithfulness | Avg Answer Length |
|--------|----------|-----------------|-------------------|
| v1 | Direct grounding | 0.998 | 370 chars |
| v2 | Chain-of-thought | 0.893 | 865 chars |

**Finding:** Concise, directly grounded prompts (v1) outperformed chain-of-thought (v2) on faithfulness. v2 produced longer answers with more sentences, giving more surface area for the NLI model to flag. This demonstrates measured prompt engineering — forming a hypothesis, testing it, and learning from the result.

## Setup
```bash
# Clone
git clone https://github.com/vk20001/news-rag.git
cd news-rag

# Environment
python3 -m venv .venv
source .venv/bin/activate

# Dependencies
pip install feedparser requests python-dotenv pyyaml
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers chromadb openai
pip install streamlit pandas pytest

# API keys (both free, no credit card)
cp .env.example .env
# Add your keys from ai.google.dev and console.groq.com
```

## Usage
```bash
# Ingest latest news
python run_ingest.py

# Chunk and embed
python run_chunk.py
python run_embed.py

# Query from CLI
python run_query.py "What is Microsoft doing about AI content?"

# Run Streamlit UI
streamlit run app.py

# Run tests
pytest tests/ -v

# Run prompt comparison
python run_prompt_comparison.py
```

## Project Structure
```
├── src/
│   ├── ingestion/       # RSS fetching, deduplication
│   ├── chunking/        # Recursive text splitting
│   ├── embedding/       # MiniLM + ChromaDB
│   ├── retrieval/       # Semantic search
│   ├── generation/      # Multi-provider LLM (Gemini/Groq)
│   ├── evaluation/      # NLI hallucination gate
│   └── monitoring/      # SQLite metrics logger
├── prompts/             # Versioned YAML prompt templates
├── tests/               # Unit tests (20 tests)
├── data/
│   ├── raw/             # Individual article JSON files
│   ├── processed/       # Chunked articles
│   ├── vectorstore/     # ChromaDB persistence
│   └── metrics.db       # Query logs
├── app.py               # Streamlit UI + dashboard
├── run_ingest.py        # Ingestion entry point
├── run_chunk.py         # Chunking entry point
├── run_embed.py         # Embedding entry point
├── run_query.py         # CLI query with logging
└── run_prompt_comparison.py  # A/B prompt experiment
```

## Monitoring Dashboard

The Streamlit dashboard tracks:
- Total queries and average faithfulness score
- Per-query breakdown with provider, prompt version, latency
- Faithfulness score distribution over time
- Refusal detection (correct "I don't know" responses)

## Built By

**Vaishnavi Kanchan** — M.Sc. Applied Data Science & Analytics, SRH Heidelberg
