"""
Two-Stage Query Router
======================

CONCEPT: Why Two Stages?
-------------------------
Stage 1 — Domain Classification (LLM call):
    Answers: "Is this query conceptually appropriate for this system?"
    Catches: completely off-topic queries, social messages, ambiguous queries
    Cannot catch: on-topic queries your KB doesn't actually cover

Stage 2 — Coverage Probe (ChromaDB query):
    Answers: "Does the knowledge base have relevant content for this query?"
    Catches: on-topic queries with no KB coverage (e.g. "OpenAI stock price")
    Cannot catch: off-topic queries (they might accidentally hit something)

CONCEPT: Four Decision Types
-----------------------------
ANSWERABLE   → full pipeline runs
OUT_OF_SCOPE → hard rejection, explain scope
AMBIGUOUS    → ask for clarification
SOCIAL       → warm acknowledgement, no retrieval needed
LOW_COVERAGE → on-topic but KB has no coverage for this specific query
"""

"""
Two-Stage Query Router with SOCIAL category support.
"""

"""
Two-Stage Query Router
======================
"""

"""
Two-Stage Query Router
======================
"""

import os
import random
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from src.routing.routing_config import (
    DOMAIN_DESCRIPTION,
    FEW_SHOT_EXAMPLES,
    COVERAGE_DISTANCE_THRESHOLD,
    COVERAGE_PROBE_TOP_K,
    GREETING_KEYWORDS,
    BOT_QUESTION_KEYWORDS,
    GREETING_RESPONSES,
    BOT_QUESTION_RESPONSES,
    ACKNOWLEDGEMENT_RESPONSES,
    OUT_OF_SCOPE_RESPONSES,
    AMBIGUOUS_RESPONSES,
    LOW_COVERAGE_RESPONSES,
)


@dataclass
class RoutingResult:
    decision: Literal["ANSWERABLE", "OUT_OF_SCOPE", "AMBIGUOUS", "LOW_COVERAGE", "SOCIAL"]
    reason: str
    query: str
    best_distance: float | None = None
    classifier_raw: str | None = None

    @property
    def should_proceed(self) -> bool:
        return self.decision == "ANSWERABLE"

    @property
    def user_message(self) -> str:
        if self.reason == "SOCIAL":
            query_lower = self.query.lower().strip().rstrip("!?.")
            if query_lower in GREETING_KEYWORDS:
                return random.choice(GREETING_RESPONSES)
            elif any(phrase in query_lower for phrase in BOT_QUESTION_KEYWORDS):
                return random.choice(BOT_QUESTION_RESPONSES)
            else:
                return random.choice(ACKNOWLEDGEMENT_RESPONSES)
        elif self.reason == "OUT_OF_SCOPE":
            return random.choice(OUT_OF_SCOPE_RESPONSES)
        elif self.reason == "AMBIGUOUS":
            return random.choice(AMBIGUOUS_RESPONSES)
        elif self.reason == "LOW_COVERAGE":
            return random.choice(LOW_COVERAGE_RESPONSES)
        return "Not sure how to help with that — try asking about recent tech news!"


def _build_classifier_prompt(query: str) -> str:
    examples_text = "\n".join(
        f'Query: "{q}" → {label}'
        for q, label in FEW_SHOT_EXAMPLES
    )

    return f"""You are a query classifier for a tech news Q&A system.
The system can only answer questions about: {DOMAIN_DESCRIPTION}.

Classify the user query into exactly one of these categories:
- ANSWERABLE: query is about tech news topics this system covers
- OUT_OF_SCOPE: query is unrelated to tech news (cooking, geography, sports, finance, etc.)
- AMBIGUOUS: query is too vague to retrieve meaningful information
- SOCIAL: greetings, thanks, acknowledgements, chitchat, pleasantries, questions about the bot

Examples:
{examples_text}

CRITICAL: Respond with ONLY the category name. No explanation. No punctuation. No other text.

Query: "{query}"
"""


def _classify_domain(query: str) -> tuple[str, str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = _build_classifier_prompt(query)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=25,
        )

        content = response.choices[0].message.content
        if content is None:
            return "ANSWERABLE", "none_response"

        raw = content.strip()

    except Exception as e:
        print(f"  [Router] Stage 1 LLM call failed: {e}")
        return "ANSWERABLE", f"classifier_error: {e}"

    normalized = raw.upper().strip()

    for label in ["OUT_OF_SCOPE", "AMBIGUOUS", "ANSWERABLE", "SOCIAL"]:
        if label in normalized:
            return label, raw

    if "OUT" in normalized or "SCOPE" in normalized:
        return "OUT_OF_SCOPE", raw
    if "AMBIG" in normalized:
        return "AMBIGUOUS", raw
    if "SOCIAL" in normalized:
        return "SOCIAL", raw
    if "ANSWER" in normalized:
        return "ANSWERABLE", raw

    print(f"  [Router] Unexpected response: '{raw}' — defaulting to ANSWERABLE")
    return "ANSWERABLE", raw


def _probe_coverage(query: str, retriever) -> tuple[bool, float]:
    try:
        probe_results = retriever.retrieve(query, top_k=COVERAGE_PROBE_TOP_K)

        if not probe_results:
            return False, 999.0

        best_distance = min(r["distance"] for r in probe_results)
        has_coverage = best_distance <= COVERAGE_DISTANCE_THRESHOLD

        print(f"  [Router] Coverage probe — best distance: {best_distance:.4f} "
              f"(threshold: {COVERAGE_DISTANCE_THRESHOLD}) → "
              f"{'PASS' if has_coverage else 'FAIL'}")

        return has_coverage, best_distance

    except Exception as e:
        print(f"  [Router] Stage 2 coverage probe failed: {e}")
        return True, 0.0


def route_query(query: str, retriever) -> RoutingResult:
    print(f"\n[Router] Classifying query: '{query}'")

    classification, raw_response = _classify_domain(query)
    print(f"  [Router] Stage 1 result: {classification} (raw: '{raw_response}')")

    if classification == "OUT_OF_SCOPE":
        return RoutingResult(decision="OUT_OF_SCOPE", reason="OUT_OF_SCOPE",
                             query=query, classifier_raw=raw_response)

    if classification == "AMBIGUOUS":
        return RoutingResult(decision="AMBIGUOUS", reason="AMBIGUOUS",
                             query=query, classifier_raw=raw_response)

    if classification == "SOCIAL":
        return RoutingResult(decision="SOCIAL", reason="SOCIAL",
                             query=query, classifier_raw=raw_response)

    has_coverage, best_distance = _probe_coverage(query, retriever)

    if not has_coverage:
        return RoutingResult(decision="LOW_COVERAGE", reason="LOW_COVERAGE",
                             query=query, best_distance=best_distance,
                             classifier_raw=raw_response)

    print(f"  [Router] Query approved for full pipeline.")
    return RoutingResult(decision="ANSWERABLE", reason="passed_both_stages",
                         query=query, best_distance=best_distance,
                         classifier_raw=raw_response)