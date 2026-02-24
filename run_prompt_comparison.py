"""
Prompt A/B Comparison: v1 vs v2
Run the same questions through both prompts and compare faithfulness.
"""
from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer
from src.evaluation.hallucination_gate import HallucinationGate

TEST_QUESTIONS = [
    "How much money has Meta lost on the metaverse?",
    "What is Microsoft doing about AI-generated content?",
    "What is the capital of France?",
    "Any news about layoffs in tech?",
    "What is happening with autonomous submarines?",
]

def main():
    retriever = Retriever()
    gate = HallucinationGate()

    results = []

    for question in TEST_QUESTIONS:
        chunks = retriever.retrieve(question, top_k=5)

        for prompt_version in ["prompts/v1.yaml", "prompts/v2.yaml"]:
            version = "v1" if "v1" in prompt_version else "v2"
            print(f"\n{'─'*50}")
            print(f"Q: {question}")
            print(f"Prompt: {version}")

            result = generate_answer(question, chunks, prompt_path=prompt_version)
            evaluation = gate.evaluate(result["answer"], chunks)

            print(f"Answer: {result['answer'][:150]}...")
            print(f"Faithfulness: {evaluation['faithfulness_score']:.2f}")
            print(f"Refusal: {evaluation.get('is_refusal', False)}")

            results.append({
                "question": question,
                "prompt": version,
                "faithfulness": evaluation["faithfulness_score"],
                "is_refusal": evaluation.get("is_refusal", False),
                "answer_length": len(result["answer"]),
            })

    # Summary
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")

    v1_scores = [r["faithfulness"] for r in results if r["prompt"] == "v1" and not r["is_refusal"]]
    v2_scores = [r["faithfulness"] for r in results if r["prompt"] == "v2" and not r["is_refusal"]]

    v1_avg = sum(v1_scores) / len(v1_scores) if v1_scores else 0
    v2_avg = sum(v2_scores) / len(v2_scores) if v2_scores else 0

    print(f"V1 avg faithfulness: {v1_avg:.4f} ({len(v1_scores)} non-refusal answers)")
    print(f"V2 avg faithfulness: {v2_avg:.4f} ({len(v2_scores)} non-refusal answers)")

    if v2_avg > v1_avg:
        print(f"V2 is better by {v2_avg - v1_avg:.4f}")
    elif v1_avg > v2_avg:
        print(f"V1 is better by {v1_avg - v2_avg:.4f}")
    else:
        print("Both prompts perform equally")

    print(f"\nPer-question breakdown:")
    for q in TEST_QUESTIONS:
        v1 = next((r for r in results if r["question"] == q and r["prompt"] == "v1"), None)
        v2 = next((r for r in results if r["question"] == q and r["prompt"] == "v2"), None)
        print(f"  {q[:45]}")
        print(f"    v1: faith={v1['faithfulness']:.2f}, len={v1['answer_length']}")
        print(f"    v2: faith={v2['faithfulness']:.2f}, len={v2['answer_length']}")

if __name__ == "__main__":
    main()
