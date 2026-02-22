"""
Hallucination Quality Gate
===========================

CONCEPT: What This Does
-----------------------
Takes Gemini's answer and checks: is this answer actually supported
by the retrieved chunks?

It uses an NLI (Natural Language Inference) model — a small classifier
that takes two texts and decides:
  - ENTAILMENT: text B is supported by text A
  - CONTRADICTION: text B conflicts with text A
  - NEUTRAL: text A doesn't say anything about text B

Example:
  Chunk (text A): "Meta has lost $80 billion on Reality Labs"
  Answer (text B): "Meta has lost $80 billion on Reality Labs investments"
  → ENTAILMENT (answer is supported by chunk)

  Chunk (text A): "Meta has lost $80 billion on Reality Labs"
  Answer (text B): "Meta has lost $120 billion on Reality Labs"
  → CONTRADICTION (answer conflicts with chunk)

  Chunk (text A): "Meta has lost $80 billion on Reality Labs"
  Answer (text B): "Mark Zuckerberg founded Facebook in 2004"
  → NEUTRAL (chunk says nothing about this — hallucination territory)

CONCEPT: How the Score Works
----------------------------
We split Gemini's answer into sentences.
For each sentence, we check it against ALL retrieved chunks.
We take the BEST entailment score across chunks
(because a sentence only needs to be supported by one chunk).

Final faithfulness score = average of best entailment scores per sentence.
- Score close to 1.0 → answer is well-supported by chunks
- Score close to 0.0 → answer is NOT supported → hallucination warning

CONCEPT: Why NLI Model and Not Ask Another LLM
-----------------------------------------------
We could ask Gemini "did you hallucinate?" But:
1. Costs API quota per query
2. LLMs are bad at evaluating themselves
3. Not deterministic — same input might get different answer

NLI model (DeBERTa, ~200MB) runs locally on CPU:
1. Free — no API calls
2. Trained specifically for entailment detection
3. Deterministic — same input always gets same score
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HallucinationGate:
    """
    Checks if an LLM answer is supported by retrieved context.
    
    Uses cross-encoder NLI model: takes (premise, hypothesis) pair
    and outputs entailment/neutral/contradiction probabilities.
    
    Premise = chunk text (what we know)
    Hypothesis = answer sentence (what the LLM claims)
    """
    
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        """
        Load the NLI model. ~200MB download on first run, cached after.
        Runs on CPU — no GPU needed.
        """
        print(f"Loading hallucination gate model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # set to evaluation mode (no training)
        
        # NLI models output 3 classes: entailment, neutral, contradiction
        # The label order depends on the model — DeBERTa NLI uses:
        # 0 = contradiction, 1 = entailment, 2 = neutral
        self.entailment_idx = 1
        self.contradiction_idx = 0
        self.neutral_idx = 2
        
        print("Hallucination gate ready.")
    
    def split_into_sentences(self, text: str) -> list[str]:
        """
        Split answer into sentences for per-sentence evaluation.
        
        Why per-sentence? Because one sentence might be supported
        while another is hallucinated. Sentence-level gives you
        granular faithfulness instead of a single blurry score.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out very short fragments and citation-only text
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def score_sentence_against_chunks(
        self, sentence: str, chunks: list[dict]
    ) -> dict:
        """
        Check one sentence against all chunks.
        
        Returns the BEST entailment score across chunks.
        A sentence only needs support from ONE chunk to be faithful.
        """
        best_score = 0.0
        best_chunk = None
        
        for chunk in chunks:
            # NLI input: (premise, hypothesis)
            # premise = chunk (what we know)
            # hypothesis = sentence (what LLM claims)
            inputs = self.tokenizer(
                chunk["text"],
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
            
            entailment_score = probs[self.entailment_idx].item()
            
            if entailment_score > best_score:
                best_score = entailment_score
                best_chunk = chunk["chunk_id"]
        
        return {
            "sentence": sentence,
            "entailment_score": round(best_score, 4),
            "best_supporting_chunk": best_chunk,
        }
    
    def evaluate(
        self, answer: str, chunks: list[dict], threshold: float = 0.5
    ) -> dict:
        """
        Main evaluation: score full answer against retrieved chunks.
        
        THRESHOLD:
        - Above threshold → answer is considered faithful
        - Below threshold → answer flagged as low-confidence
        
        0.5 is a starting point. You'll tune this based on
        your RAGAS evaluation results later.
        
        Returns:
        - faithfulness_score: average entailment across sentences
        - is_faithful: whether score meets threshold
        - sentence_scores: per-sentence breakdown
        - flagged_sentences: sentences below threshold
        """
        sentences = self.split_into_sentences(answer)
        
        if not sentences:
            return {
                "faithfulness_score": 0.0,
                "is_faithful": False,
                "sentence_scores": [],
                "flagged_sentences": [],
                "num_sentences": 0,
            }
        
        sentence_scores = []
        for sentence in sentences:
            score = self.score_sentence_against_chunks(sentence, chunks)
            sentence_scores.append(score)
        
        # Overall faithfulness = average of best entailment scores
        avg_score = sum(s["entailment_score"] for s in sentence_scores) / len(sentence_scores)
        
        # Flag sentences below threshold
        flagged = [s for s in sentence_scores if s["entailment_score"] < threshold]
        
        return {
            "faithfulness_score": round(avg_score, 4),
            "is_faithful": avg_score >= threshold,
            "sentence_scores": sentence_scores,
            "flagged_sentences": flagged,
            "num_sentences": len(sentences),
            "threshold": threshold,
        }
