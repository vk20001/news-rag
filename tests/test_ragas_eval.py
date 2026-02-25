"""
RAGAS Evaluation Suite
======================
Measures RAG pipeline quality using 2 metrics:
- Faithfulness: does the answer contain only claims supported by context?
- Answer Relevancy: does the answer address the question?

Run locally:
    pytest tests/test_ragas_eval.py -v -m ragas

Skipped in normal pytest runs — only triggered explicitly in CI.
"""
import json
import os
import pytest

pytestmark = pytest.mark.ragas


def test_ragas_metrics():
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import faithfulness, answer_relevancy
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from src.retrieval.retriever import Retriever
    from src.generation.generator import generate_answer

    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path) as f:
        eval_questions = json.load(f)

    retriever = Retriever()

    questions, answers, contexts = [], [], []

    for item in eval_questions:
        q = item["question"]
        chunks = retriever.retrieve(q, top_k=5)
        context_texts = [c["text"] for c in chunks]
        result = generate_answer(q, chunks)
        answer = result["answer"]

        questions.append(q)
        answers.append(answer)
        contexts.append(context_texts)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print(f"\nRAGAS Results:")
    print(f"  Faithfulness:      {results['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {results['answer_relevancy']:.3f}")

    assert results["faithfulness"] >= 0.7, \
        f"Faithfulness too low: {results['faithfulness']:.3f}"
    assert results["answer_relevancy"] >= 0.6, \
        f"Answer relevancy too low: {results['answer_relevancy']:.3f}"