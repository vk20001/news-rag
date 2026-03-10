"""
Tech News Assistant — Conversational RAG
"""

import streamlit as st
import time
from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer
from src.generation.query_rewriter import rewrite_query
from src.evaluation.hallucination_gate import HallucinationGate
from src.monitoring.metrics_logger import MetricsLogger
from src.routing.query_router import route_query


st.set_page_config(
    page_title="Tech News Assistant",
    page_icon="📰",
    layout="centered",
)


@st.cache_resource
def load_retriever():
    return Retriever()

@st.cache_resource
def load_gate():
    return HallucinationGate()

@st.cache_resource
def load_logger():
    return MetricsLogger()


if "messages" not in st.session_state:
    st.session_state.messages = []


def query_page():
    st.title("📰 Tech News Assistant")
    st.caption("Ask me anything about recent tech news. I remember our conversation.")

    retriever = load_retriever()
    gate = load_gate()
    logger = load_logger()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Ask me about tech news..."):

        with st.chat_message("user"):
            st.write(query)

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            start_time = time.time()

            # ── Step 1: Route the RAW query first ──
            # Check for SOCIAL/AMBIGUOUS BEFORE rewriting.
            # If we rewrite first, "hey" gets rewritten into a tech query
            # using conversation context — totally wrong behaviour.
            with st.spinner(""):
                raw_routing = route_query(query, retriever)

            if raw_routing.decision in ("SOCIAL", "AMBIGUOUS"):
                response = raw_routing.user_message
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                latency = time.time() - start_time
                logger.log_query(
                    query=query,
                    answer=response,
                    provider="router",
                    model="groq/llama-3.3-70b-versatile",
                    prompt_version="routing_v1",
                    chunks=[],
                    evaluation={
                        "is_faithful": False,
                        "faithfulness_score": 0.0,
                        "is_refusal": True,
                        "routing_decision": raw_routing.decision,
                        "best_distance": None,
                    },
                    latency_seconds=latency,
                )
                st.stop()

            # ── Step 2: Rewrite query using conversation context ──
            # Only runs for potentially ANSWERABLE queries — not social/ambiguous
            with st.spinner(""):
                rewritten_query = rewrite_query(
                    query=query,
                    conversation_history=st.session_state.messages[:-1],
                )

            # ── Step 3: Route the REWRITTEN query ──
            # Now check OUT_OF_SCOPE + LOW_COVERAGE on the resolved query
            with st.spinner(""):
                routing = route_query(rewritten_query, retriever)

            if not routing.should_proceed:
                response = routing.user_message
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                latency = time.time() - start_time
                logger.log_query(
                    query=query,
                    answer=response,
                    provider="router",
                    model="groq/llama-3.3-70b-versatile",
                    prompt_version="routing_v1",
                    chunks=[],
                    evaluation={
                        "is_faithful": False,
                        "faithfulness_score": 0.0,
                        "is_refusal": True,
                        "routing_decision": routing.decision,
                        "best_distance": routing.best_distance,
                    },
                    latency_seconds=latency,
                )
                st.stop()

            # ── Step 4: Retrieve ──
            with st.spinner("Searching articles..."):
                chunks = retriever.retrieve(rewritten_query, top_k=5)

            # ── Step 5: Generate ──
            with st.spinner("On it..."):
                result = generate_answer(
                    query=rewritten_query,
                    retrieved_chunks=chunks,
                    conversation_history=st.session_state.messages[:-1],
                )

            # ── Step 6: Hallucination check (silent) ──
            evaluation = gate.evaluate(result["answer"], chunks)
            evaluation["routing_decision"] = routing.decision
            evaluation["best_distance"] = routing.best_distance

            latency = time.time() - start_time

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

            st.write(result["answer"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
            })

    with st.sidebar:
        st.markdown("### Tech News Assistant")
        st.caption("Powered by RAG + NLI hallucination detection")
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


def dashboard_page():
    import pandas as pd

    st.title("Pipeline Monitoring Dashboard")

    if st.sidebar.button("← Back to Chat", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()

    logger = load_logger()
    stats = logger.get_summary_stats()

    if stats["total_queries"] == 0:
        st.info("No queries logged yet.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", stats["total_queries"])
    col2.metric("Avg Faithfulness", f"{stats['avg_faithfulness']:.2f}")
    col3.metric("Avg Latency", f"{stats['avg_latency']:.1f}s")
    col4.metric("Refusals", stats["refusal_count"])

    st.divider()

    logs = logger.get_recent_logs(20)
    if logs:
        st.subheader("Recent Queries")
        df = pd.DataFrame(logs)
        display_cols = ["timestamp", "query", "faithfulness_score", "is_faithful", "provider", "latency_seconds"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Faithfulness Scores")
        scores = [l["faithfulness_score"] for l in logs if l["faithfulness_score"] is not None]
        if scores:
            st.bar_chart(pd.DataFrame({"Faithfulness Score": scores}))


if "page" not in st.session_state:
    st.session_state.page = "chat"

if st.session_state.page == "chat":
    query_page()
else:
    dashboard_page()