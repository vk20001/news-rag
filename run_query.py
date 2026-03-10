"""
Full RAG Pipeline: Route → Retrieve → Generate → Verify → Log
                   ^^^^^^
                   NEW: two-stage router sits before everything else
"""
import sys
import time
from src.retrieval.retriever import Retriever
from src.routing.query_router import route_query
from src.generation.generator import generate_answer
from src.evaluation.hallucination_gate import HallucinationGate
from src.monitoring.metrics_logger import MetricsLogger


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_query.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    start_time = time.time()

    # Load retriever once — router and pipeline both use it
    # This is why route_query() takes a retriever argument:
    # we don't want to load the embedding model twice
    retriever = Retriever()

    # ------------------------------------------------------------------
    # TWO-STAGE ROUTING — happens before any retrieval or generation
    # ------------------------------------------------------------------
    routing = route_query(query, retriever)

    if not routing.should_proceed:
        # Query rejected — log it and exit cleanly
        print(f"\n[ROUTER DECISION] {routing.decision}")
        print(f"{routing.user_message}")

        # Log the routing rejection to SQLite so dashboard shows it
        latency = time.time() - start_time
        logger = MetricsLogger()
        logger.log_query(
            query=query,
            answer=routing.user_message,
            provider="router",
            model="gemini-2.5-flash-classifier",
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
        print(f"\nRouting rejection logged. Latency: {latency:.1f}s")
        sys.exit(0)

    # ------------------------------------------------------------------
    # FULL PIPELINE — only runs if routing approved the query
    # ------------------------------------------------------------------
    # Note: retriever is already initialized above, reuse it
    # The router already did a lightweight probe (top-3)
    # Now we do the full retrieval (top-5) for generation
    chunks = retriever.retrieve(query, top_k=5)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(chunks)} chunks (best distance: {chunks[0]['distance']:.4f})")

    print("Generating answer...")
    result = generate_answer(query, chunks)
    answer = result["answer"]

    print(f"\n{'='*50}")
    print(f"ANSWER (via {result['provider']}/{result['model']}, prompt {result['prompt_version']})")
    print(f"{'='*50}")
    print(answer)

    print(f"\n{'='*50}")
    print("HALLUCINATION CHECK")
    print(f"{'='*50}")

    gate = HallucinationGate()
    evaluation = gate.evaluate(answer, chunks)

    # Add routing metadata to evaluation for logging
    evaluation["routing_decision"] = routing.decision
    evaluation["best_distance"] = routing.best_distance

    if evaluation.get("is_refusal"):
        print("LLM correctly refused — no relevant context found.")
        print("Faithfulness: PASSED (refusal is correct behavior)")
    elif evaluation["is_faithful"]:
        print(f"Faithfulness Score: {evaluation['faithfulness_score']:.2f} — PASSED")
        for s in evaluation["sentence_scores"]:
            print(f"  [OK] ({s['entailment_score']:.2f}) {s['sentence'][:80]}...")
    else:
        print(f"Faithfulness Score: {evaluation['faithfulness_score']:.2f} — LOW CONFIDENCE")
        for s in evaluation["sentence_scores"]:
            status = "OK" if s["entailment_score"] >= evaluation["threshold"] else "FLAGGED"
            print(f"  [{status}] ({s['entailment_score']:.2f}) {s['sentence'][:80]}...")

    latency = time.time() - start_time
    logger = MetricsLogger()
    logger.log_query(
        query=query,
        answer=answer,
        provider=result["provider"],
        model=result["model"],
        prompt_version=result["prompt_version"],
        chunks=chunks,
        evaluation=evaluation,
        latency_seconds=latency,
    )
    print(f"\nMetrics logged. Latency: {latency:.1f}s")

    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']}: {s['title']}")


if __name__ == "__main__":
    main()