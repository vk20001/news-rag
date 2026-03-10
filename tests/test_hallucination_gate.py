"""Tests for the hallucination gate."""
import pytest
from src.evaluation.hallucination_gate import HallucinationGate


@pytest.fixture(scope="module")
def gate():
    """Load NLI model once for all tests in this module."""
    return HallucinationGate()


def test_clean_answer_removes_citations(gate):
    text = "Meta lost $80 billion [Source 1]."
    cleaned = gate.clean_answer(text)
    assert "[Source" not in cleaned


def test_clean_answer_no_citations(gate):
    text = "Meta lost $80 billion."
    assert gate.clean_answer(text) == text


def test_is_refusal_detects_refusal(gate):
    assert gate.is_refusal("I don't have enough information in my sources to answer this.")


def test_is_refusal_not_refusal(gate):
    assert not gate.is_refusal("Meta lost $80 billion on Reality Labs.")


def test_split_into_sentences(gate):
    text = "First sentence. Second sentence. Third sentence."
    sentences = gate.split_into_sentences(text)
    assert len(sentences) == 3


def test_split_filters_short_fragments(gate):
    text = "OK. This is a real sentence with enough length."
    sentences = gate.split_into_sentences(text)
    assert len(sentences) == 1


def test_evaluate_refusal(gate):
    result = gate.evaluate("I don't have enough information in my sources to answer this.", [])
    assert result["is_refusal"] == True
    assert result["faithfulness_score"] == 1.0