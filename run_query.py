"""
Full RAG Pipeline: Retrieve → Generate → Verify → Log
"""
import sys
import time
from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer
from src.evaluation.hallucination_gate import HallucinationGate
from src.monitoring.metrics_logger import MetricsLogger


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_query.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    start_time = time.time()

    retriever = Retriever()
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

    # Log metrics
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
