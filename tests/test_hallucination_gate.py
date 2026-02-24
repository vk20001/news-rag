"""Tests for the hallucination gate."""
from src.evaluation.hallucination_gate import HallucinationGate


gate = HallucinationGate()

def test_clean_answer_removes_citations():
    text = "Meta lost $80 billion [Source 1]."
    cleaned = gate.clean_answer(text)
    assert "[Source" not in cleaned

def test_clean_answer_no_citations():
    text = "Meta lost $80 billion."
    assert gate.clean_answer(text) == text

def test_is_refusal_detects_refusal():
    assert gate.is_refusal("I don't have enough information in my sources to answer this.")

def test_is_refusal_not_refusal():
    assert not gate.is_refusal("Meta lost $80 billion on Reality Labs.")

def test_split_into_sentences():
    text = "First sentence. Second sentence. Third sentence."
    sentences = gate.split_into_sentences(text)
    assert len(sentences) == 3

def test_split_filters_short_fragments():
    text = "OK. This is a real sentence with enough length."
    sentences = gate.split_into_sentences(text)
    assert len(sentences) == 1  # "OK." is too short

def test_evaluate_refusal():
    result = gate.evaluate("I don't have enough information in my sources to answer this.", [])
    assert result["is_refusal"] == True
    assert result["faithfulness_score"] == 1.0
