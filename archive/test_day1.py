"""
Day 1 Test — search_semantic_scholar()

Run this file directly to verify the function works:
    python test_day1.py

What this script does:
    1. Calls search_semantic_scholar() with 3 different queries
    2. Prints the raw output for each so you can inspect the data structure
    3. Verifies the shape: every paper should have all 7 required keys

This is NOT a unit test framework — just a manual verification script.

NOTE ON RATE LIMITS:
    Semantic Scholar allows 1 request/second without an API key.
    We add a 2-second gap between queries to stay within the limit.
    Get a free API key at: https://www.semanticscholar.org/product/api
    (With a key you get 100 requests/second instead of 1.)
"""

import time
from utils.semantic_scholar import search_semantic_scholar

# Seconds to wait between queries — keeps us inside the unauthenticated rate limit
DELAY_BETWEEN_QUERIES = 2

# ── 3 test queries ─────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "transformers for natural language processing",
    "LSTM time series forecasting",
    "attention mechanisms in deep learning",
]

# These are the 7 keys your spec requires every paper to have.
REQUIRED_KEYS = {"paper_id", "title", "abstract", "authors", "year", "citation_count", "url"}


def print_paper(paper: dict, index: int):
    """Print one paper's data in a readable format."""
    print(f"\n  Paper {index + 1}:")
    print(f"    paper_id      : {paper['paper_id']}")
    print(f"    title         : {paper['title']}")
    print(f"    year          : {paper['year']}")
    print(f"    citation_count: {paper['citation_count']}")
    print(f"    authors       : {paper['authors'][:3]}{'...' if len(paper['authors']) > 3 else ''}")
    print(f"    url           : {paper['url']}")
    # Print only the first 120 characters of the abstract to keep output readable
    abstract_preview = (paper['abstract'][:120] + "...") if len(paper['abstract']) > 120 else paper['abstract']
    print(f"    abstract      : {abstract_preview!r}")


def verify_shape(papers: list[dict], query: str) -> bool:
    """Check that every paper has all 7 required keys. Returns True if all good."""
    for i, paper in enumerate(papers):
        missing = REQUIRED_KEYS - paper.keys()
        if missing:
            print(f"  ❌ Paper {i+1} is missing keys: {missing}")
            return False
    print(f"  ✅ All {len(papers)} papers have the correct 7 keys.")
    return True


# ── Run the tests ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Day 1 Test: search_semantic_scholar()")
    print("=" * 60)

    for idx, query in enumerate(TEST_QUERIES):
        print(f"\n{'─' * 60}")
        print(f"Query: '{query}'")
        print("─" * 60)

        papers = search_semantic_scholar(query, limit=8)

        if not papers:
            print("  ⚠️  No papers returned — check the error printed above.")
        else:
            print(f"  Returned {len(papers)} papers.")
            verify_shape(papers, query)
            for i, paper in enumerate(papers):
                print_paper(paper, i)

        # Wait between queries so we don't hit the unauthenticated rate limit
        if idx < len(TEST_QUERIES) - 1:
            print(f"\n  (waiting {DELAY_BETWEEN_QUERIES}s before next query…)")
            time.sleep(DELAY_BETWEEN_QUERIES)

    print(f"\n{'=' * 60}")
    print("Test complete.")
    print("=" * 60)
