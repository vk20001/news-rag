"""
LLM Generation Module
=====================

CONCEPT: This is Where the LLM Reads the Cheat Sheet and Answers
-----------------------------------------------------------------
The retriever found the most relevant chunks for the user's question.
Now we stuff those chunks into a prompt and send it to Gemini.

CONCEPT: The Prompt is Everything
---------------------------------
The prompt tells the LLM:
1. What role it plays (a news Q&A assistant)
2. What context it has (the retrieved chunks)
3. What rules to follow (answer ONLY from context, cite sources)
4. What question to answer

How you word this prompt DRAMATICALLY affects output quality.
"Answer the question" vs "Answer the question using ONLY the provided
context. If the context doesn't contain the answer, say so." —
the second version halves hallucination rate.

This is prompt engineering. You'll experiment with different wordings
and measure which produces better faithfulness scores.

CONCEPT: OpenAI-Compatible SDK
-------------------------------
Gemini and Groq both support the OpenAI SDK format.
We use the `openai` Python library but point it at Gemini's URL.
This means switching providers = changing 2 lines (base_url + api_key).
Real production systems do this for vendor flexibility.

CONCEPT: Prompt Versioning (YAML)
---------------------------------
We store prompts as YAML files in prompts/ directory.
Each has a version number. Every response logs which version produced it.
Later you can compare: "Did prompt v2 produce fewer hallucinations than v1?"
This is how real teams iterate on prompts — measured, not vibes.
"""

import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_prompt_template(prompt_path: str = "prompts/v1.yaml") -> dict:
    """
    Load a versioned prompt template from YAML.
    
    The YAML file contains:
    - version: for tracking
    - system_prompt: instructions to the LLM
    - user_template: how to format the question + context
    """
    with open(prompt_path, "r") as f:
        return yaml.safe_load(f)


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a string for the prompt.
    
    Each chunk gets a number and source attribution.
    This helps the LLM cite sources in its answer.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']} — {chunk['title']}]\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def get_llm_client(provider: str = "gemini") -> tuple[OpenAI, str]:
    """
    Create an OpenAI-compatible client for Gemini or Groq.
    
    Both providers support the OpenAI SDK format.
    Same code, different base_url and api_key. That's it.
    
    Returns (client, model_name) tuple.
    """
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = "gemini-2.5-flash"
        return client, model
    
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = "llama-3.3-70b-versatile"
        return client, model
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    provider: str = "gemini",
    prompt_path: str = "prompts/v1.yaml",
) -> dict:
    """
    Generate an answer using the LLM with retrieved context.
    
    FLOW:
    1. Load prompt template (versioned YAML)
    2. Format retrieved chunks into context string
    3. Build the full prompt (system + user message)
    4. Send to LLM API
    5. Return answer + metadata for logging
    
    FALLBACK LOGIC:
    If primary provider (Gemini) fails, try fallback (Groq).
    This is a production pattern — never depend on a single API.
    """
    # Load prompt template
    prompt_config = load_prompt_template(prompt_path)
    
    # Format context from retrieved chunks
    context = format_context(retrieved_chunks)
    
    # Build user message from template
    user_message = prompt_config["user_template"].format(
        context=context,
        question=query,
    )
    
    # Try primary provider, fall back if it fails
    providers_to_try = [provider]
    fallback = "groq" if provider == "gemini" else "gemini"
    providers_to_try.append(fallback)
    
    last_error = None
    
    for p in providers_to_try:
        try:
            client, model = get_llm_client(p)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt_config["system_prompt"]},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,  # low temperature = more factual, less creative
                max_tokens=1024,
            )
            
            answer = response.choices[0].message.content
            
            return {
                "query": query,
                "answer": answer,
                "provider": p,
                "model": model,
                "prompt_version": prompt_config["version"],
                "context_chunks": len(retrieved_chunks),
                "sources": [
                    {"source": c["source"], "title": c["title"], "url": c["url"]}
                    for c in retrieved_chunks
                ],
            }
        
        except Exception as e:
            print(f"  {p} failed: {e}")
            last_error = e
            continue
    
    # Both providers failed
    return {
        "query": query,
        "answer": f"Error: All providers failed. Last error: {last_error}",
        "provider": "none",
        "model": "none",
        "prompt_version": prompt_config.get("version", "unknown"),
        "context_chunks": len(retrieved_chunks),
        "sources": [],
    }
