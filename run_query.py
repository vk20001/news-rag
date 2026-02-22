"""
Full RAG Pipeline: Retrieve → Generate → Verify
Run AFTER run_embed.py.

Usage:
    python run_query.py "your question here"
"""
import sys
from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer
from src.evaluation.hallucination_gate import HallucinationGate


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_query.py \"your question here\"")
        sys.exit(1)
    
    query = sys.argv[1]
    
    # Step 1: Retrieve
    retriever = Retriever()
    chunks = retriever.retrieve(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(chunks)} chunks (best distance: {chunks[0]['distance']:.4f})")
    
    # Step 2: Generate
    print("Generating answer...")
    result = generate_answer(query, chunks)
    
    print(f"\n{'='*50}")
    print(f"ANSWER (via {result['provider']}/{result['model']}, prompt {result['prompt_version']})")
    print(f"{'='*50}")
    print(result["answer"])
    
    # Step 3: Hallucination Gate
    print(f"\n{'='*50}")
    print("HALLUCINATION CHECK")
    print(f"{'='*50}")
    
    gate = HallucinationGate()
    evaluation = gate.evaluate(result["answer"], chunks)
    
    score = evaluation["faithfulness_score"]
    
    if evaluation["is_faithful"]:
        print(f"Faithfulness Score: {score:.2f} — PASSED")
    else:
        print(f"Faithfulness Score: {score:.2f} — LOW CONFIDENCE")
        print("WARNING: This answer may contain information not supported by sources.")
    
    print(f"\nPer-sentence breakdown:")
    for s in evaluation["sentence_scores"]:
        status = "OK" if s["entailment_score"] >= evaluation["threshold"] else "FLAGGED"
        print(f"  [{status}] ({s['entailment_score']:.2f}) {s['sentence'][:80]}...")
    
    if evaluation["flagged_sentences"]:
        print(f"\n{len(evaluation['flagged_sentences'])} sentence(s) flagged as potentially unsupported.")
    
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']}: {s['title']}")

if __name__ == "__main__":
    main()
