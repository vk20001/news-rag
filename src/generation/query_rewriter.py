"""
Query Rewriter
==============

CONCEPT: Why Rewriting Is Needed
----------------------------------
Retrieval works on the EXACT query you give it.
"What about their funding?" embeds as a vague, context-less vector.
The embedding model has no idea what "their" refers to.

The rewriter resolves this by using conversation history to turn
vague follow-up questions into standalone, retrievable queries.

"What about their funding?" + [OpenAI context] 
→ "What is OpenAI's latest funding situation?"

CONCEPT: Why Groq, Not Gemini
------------------------------
Same reason as the classifier — this is a cheap utility call.
Short input, short output, needs to be fast.
Groq handles this in ~0.5s and doesn't eat Gemini quota.
Gemini is reserved for the expensive generation step.

CONCEPT: Last N Turns, Not Full History
-----------------------------------------
We only pass the last 3 conversation turns to the rewriter.
Full history would:
- Cost more tokens per rewrite call
- Confuse the rewriter with old, irrelevant context
- Make no practical difference (turn 6 rarely depends on turn 1)
3 turns captures enough context for all realistic coreference cases.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# How many previous turns to include in rewrite context
# Each "turn" = one user message + one assistant response
REWRITE_CONTEXT_TURNS = 3


def _get_groq_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def rewrite_query(query: str, conversation_history: list[dict]) -> str:
    """
    Rewrite a potentially vague query into a standalone question
    using recent conversation history for context.

    Args:
        query: the current user message (may be vague/referential)
        conversation_history: list of {"role": "user"/"assistant", "content": "..."}
                              This is the FULL history — we slice it internally.

    Returns:
        A standalone query string safe for retrieval.
        If no history or query is already clear, returns query unchanged.

    CONCEPT: Fail-Safe Design
    --------------------------
    If the rewriter API call fails for any reason (quota, network),
    we return the original query unchanged. The pipeline continues —
    retrieval quality may suffer slightly but the system doesn't crash.
    This is called "graceful degradation."
    """
    # First message — no history to use, skip rewriting
    if not conversation_history:
        return query

    # Slice to last N turns (each turn = 2 messages: user + assistant)
    recent_history = conversation_history[-(REWRITE_CONTEXT_TURNS * 2):]

    # Format history as readable context for the rewriter prompt
    history_text = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant responses — we only need enough context
        # for coreference resolution, not the full answer
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        history_text += f"{role}: {content}\n"

    prompt = f"""You are a query rewriter for a tech news Q&A system.

Given a conversation history and a follow-up question, rewrite the follow-up 
question as a complete, standalone question that can be understood without 
the conversation history.

Rules:
- If the question is already complete and unambiguous, return it UNCHANGED
- Resolve pronouns and references (their, it, this, that, the company, etc.)
- Keep the rewritten question concise — one sentence maximum
- Do NOT answer the question, only rewrite it
- Return ONLY the rewritten question, no explanation, no punctuation changes

Conversation history:
{history_text}
Follow-up question: {query}

Rewritten question:"""

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,  # rewritten query should be short
        )

        content = response.choices[0].message.content
        if content is None:
            return query

        rewritten = content.strip()

        # Sanity check — if rewriter returns something very long or empty,
        # it misunderstood the task. Fall back to original.
        if not rewritten or len(rewritten) > 300:
            return query

        print(f"  [Rewriter] '{query}' → '{rewritten}'")
        return rewritten

    except Exception as e:
        print(f"  [Rewriter] Failed, using original query: {e}")
        return query  # graceful degradation
