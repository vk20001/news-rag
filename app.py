"""
Streamlit UI for RAG Quality Pipeline
======================================

Two pages:
1. Query — ask questions, see answers with faithfulness scores
2. Dashboard — monitor system performance over time
"""

import streamlit as st
import time
import pandas as pd
from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer
from src.evaluation.hallucination_gate import HallucinationGate
from src.monitoring.metrics_logger import MetricsLogger


# ──────────────────────────────────────────────
# Load models ONCE — cached in session state
# This is why we used a class for Retriever and HallucinationGate.
# Without caching, models reload on every interaction (36s each time).
# With caching, they load once at startup and stay in memory.
# ──────────────────────────────────────────────

@st.cache_resource
def load_retriever():
    return Retriever()

@st.cache_resource
def load_gate():
    return HallucinationGate()

@st.cache_resource
def load_logger():
    return MetricsLogger()


def query_page():
    """Main query interface."""
    st.title("Tech News RAG Pipeline")
    st.caption("Ask questions about recent tech news — answers grounded in real sources")

    query = st.text_input("Your question:", placeholder="e.g. What is Microsoft doing about AI content?")

    if st.button("Ask", type="primary") and query:
        retriever = load_retriever()
        gate = load_gate()
        logger = load_logger()

        start_time = time.time()

        # Step 1: Retrieve
        with st.spinner("Searching articles..."):
            chunks = retriever.retrieve(query, top_k=5)

        # Step 2: Generate
        with st.spinner("Generating answer..."):
            result = generate_answer(query, chunks)

        # Step 3: Hallucination check
        with st.spinner("Checking faithfulness..."):
            evaluation = gate.evaluate(result["answer"], chunks)

        latency = time.time() - start_time

        # Log metrics
        logger.log_query(
            query=query,
            answer=result["answer"],
            provider=result["provider"],
            model=result["model"],
            prompt_version=result["prompt_version"],
            chunks=chunks,
            evaluation=evaluation,
            latency_seconds=latency,
        )

        # ── Display Answer ──
        st.markdown("### Answer")
        st.write(result["answer"])

        # ── Faithfulness Badge ──
        if evaluation.get("is_refusal"):
            st.info("LLM correctly refused — no relevant context in sources.")
        elif evaluation["is_faithful"]:
            st.success(f"Faithfulness: {evaluation['faithfulness_score']:.2f} — Verified")
        else:
            st.warning(f"Faithfulness: {evaluation['faithfulness_score']:.2f} — Low confidence")
            if evaluation.get("flagged_sentences"):
                with st.expander("Flagged sentences"):
                    for s in evaluation["flagged_sentences"]:
                        st.write(f"- ({s['entailment_score']:.2f}) {s['sentence']}")

        # ── Sentence Breakdown ──
        if evaluation.get("sentence_scores"):
            with st.expander("Per-sentence faithfulness breakdown"):
                for s in evaluation["sentence_scores"]:
                    score = s["entailment_score"]
                    icon = "✅" if score >= 0.5 else "⚠️"
                    st.write(f"{icon} ({score:.2f}) {s['sentence']}")

        # ── Sources ──
        with st.expander("Sources"):
            seen = set()
            for s in result["sources"]:
                key = s["url"]
                if key not in seen:
                    seen.add(key)
                    st.write(f"**{s['source']}**: [{s['title']}]({s['url']})")

        # ── Metadata ──
        col1, col2, col3 = st.columns(3)
        col1.metric("Provider", f"{result['provider']}")
        col2.metric("Prompt Version", result["prompt_version"])
        col3.metric("Latency", f"{latency:.1f}s")


def dashboard_page():
    """Monitoring dashboard."""
    st.title("Pipeline Monitoring Dashboard")

    logger = load_logger()
    stats = logger.get_summary_stats()

    if stats["total_queries"] == 0:
        st.info("No queries logged yet. Go to the Query page and ask some questions.")
        return

    # ── Summary Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", stats["total_queries"])
    col2.metric("Avg Faithfulness", f"{stats['avg_faithfulness']:.2f}")
    col3.metric("Avg Latency", f"{stats['avg_latency']:.1f}s")
    col4.metric("Refusals", stats["refusal_count"])

    st.divider()

    # ── Recent Queries Table ──
    st.subheader("Recent Queries")
    logs = logger.get_recent_logs(20)

    if logs:
        df = pd.DataFrame(logs)
        display_cols = ["timestamp", "query", "faithfulness_score", "is_faithful", "is_refusal", "provider", "prompt_version", "latency_seconds"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[available],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── Faithfulness Distribution ──
    if logs:
        st.subheader("Faithfulness Scores")
        scores = [l["faithfulness_score"] for l in logs if l["faithfulness_score"] is not None]
        if scores:
            chart_data = pd.DataFrame({"Faithfulness Score": scores})
            st.bar_chart(chart_data)


# ──────────────────────────────────────────────
# Page Navigation
# ──────────────────────────────────────────────

page = st.sidebar.radio("Navigate", ["Query", "Dashboard"])

if page == "Query":
    query_page()
else:
    dashboard_page()
