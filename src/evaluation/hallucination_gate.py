"""
Hallucination Quality Gate
===========================
Uses CrossEncoder NLI model to check if LLM answers are supported
by retrieved chunks. Refusal answers are handled separately as
correct behavior — not scored through NLI.
"""

import re
import numpy as np
from sentence_transformers import CrossEncoder


class HallucinationGate:

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        print(f"Loading hallucination gate model: {model_name}")
        self.model = CrossEncoder(model_name)
        # CrossEncoder predict() output order (verified empirically):
        # Index 0 = contradiction, Index 1 = neutral, Index 2 = entailment
        self.entailment_idx = 2
        self.contradiction_idx = 0
        self.neutral_idx = 1
        print("Hallucination gate ready.")

    def clean_answer(self, text: str) -> str:
        return re.sub(r"\[Source \d+[^\]]*\]", "", text).strip()

    def is_refusal(self, text: str) -> bool:
        refusal_patterns = [
            "i don't have enough information",
            "not enough information in my sources",
            "cannot answer",
            "no relevant information",
            "context does not contain",
            "sources do not contain",
        ]
        lower = text.lower()
        return any(p in lower for p in refusal_patterns)

    def split_into_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def score_sentence_against_chunks(self, sentence: str, chunks: list[dict]) -> dict:
        best_score = 0.0
        best_chunk = None
        pairs = [(chunk["text"], sentence) for chunk in chunks]
        raw_scores = self.model.predict(pairs)
        for i, scores in enumerate(raw_scores):
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            entailment_score = float(probs[self.entailment_idx])
            if entailment_score > best_score:
                best_score = entailment_score
                best_chunk = chunks[i]["chunk_id"]
        return {
            "sentence": sentence,
            "entailment_score": round(best_score, 4),
            "best_supporting_chunk": best_chunk,
        }

    def evaluate(self, answer: str, chunks: list[dict], threshold: float = 0.5) -> dict:
        cleaned = self.clean_answer(answer)
        if self.is_refusal(cleaned):
            return {
                "faithfulness_score": 1.0,
                "is_faithful": True,
                "is_refusal": True,
                "sentence_scores": [],
                "flagged_sentences": [],
                "num_sentences": 0,
            }
        sentences = self.split_into_sentences(cleaned)
        if not sentences:
            return {
                "faithfulness_score": 0.0,
                "is_faithful": False,
                "is_refusal": False,
                "sentence_scores": [],
                "flagged_sentences": [],
                "num_sentences": 0,
            }
        sentence_scores = []
        for sentence in sentences:
            score = self.score_sentence_against_chunks(sentence, chunks)
            sentence_scores.append(score)
        avg_score = sum(s["entailment_score"] for s in sentence_scores) / len(sentence_scores)
        flagged = [s for s in sentence_scores if s["entailment_score"] < threshold]
        return {
            "faithfulness_score": round(avg_score, 4),
            "is_faithful": avg_score >= threshold,
            "is_refusal": False,
            "sentence_scores": sentence_scores,
            "flagged_sentences": flagged,
            "num_sentences": len(sentences),
            "threshold": threshold,
        }
