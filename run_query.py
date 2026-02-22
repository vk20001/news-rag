"""
CLI entry point for querying the RAG pipeline.
Run AFTER run_embed.py.

Usage:
    python run_query.py "your question here"
"""
import sys
from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_query.py \"your question here\"")
        sys.exit(1)
    
    query = sys.argv[1]
    
    # Step 1: Retrieve relevant chunks
    retriever = Retriever()
    chunks = retriever.retrieve(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(chunks)} chunks")
    print(f"Best match distance: {chunks[0]['distance']:.4f}")
    print(f"Best match source: {chunks[0]['source']} — {chunks[0]['title']}")
    
    # Step 2: Generate answer using LLM
    print(f"\nGenerating answer...")
    result = generate_answer(query, chunks)
    
    print(f"\n{'='*50}")
    print(f"ANSWER (via {result['provider']}/{result['model']}, prompt {result['prompt_version']})")
    print(f"{'='*50}")
    print(result["answer"])
    print(f"\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['source']}: {s['title']}")
        print(f"    {s['url']}")

if __name__ == "__main__":
    main()
