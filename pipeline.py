"""
pipeline.py — the orchestrator

This file imports all 5 agents and runs them in sequence.
It is the single function called by both:
    - app.py       (the Streamlit UI on Day 5)
    - evaluate.py  (the evaluation script on Day 6)

Having one central pipeline function means: if you change the agent order or
add a new agent, you change it here and both the UI and evaluation pick it up
automatically.

Usage:
    from pipeline import run_pipeline
    markdown, report_dict = run_pipeline("transformers for NLP")
"""

from agents.query_expander import QueryExpander
from agents.search_agent import SearchAgent
from agents.ranker_agent import RankerAgent
from agents.summarizer_agent import SummarizerAgent
from agents.report_compiler import ReportCompiler

# Create one instance of each agent.
# We create them at module level (outside the function) so they are
# instantiated once when pipeline.py is imported, not on every call.
_query_expander  = QueryExpander()
_search_agent    = SearchAgent()
_ranker_agent    = RankerAgent()
_summarizer_agent = SummarizerAgent()
_report_compiler = ReportCompiler()


def run_pipeline(query: str, status_callback=None) -> tuple[str, dict]:
    """
    Run all 5 agents in sequence and return the final report.

    Args:
        query           — the user's raw search topic
        status_callback — optional function(message: str) called before each
                          agent step. Used by the Streamlit UI to show progress.
                          Pass None (default) to run silently.

    Returns:
        (markdown_string, report_dict)
        markdown_string — formatted report ready to display in Streamlit
        report_dict     — same data as a Python dict, serializable to JSON
    """

    def update(message: str):
        """Call the status callback if one was provided, otherwise do nothing."""
        if status_callback:
            status_callback(message)

    # ── Agent 1: Query Expander ────────────────────────────────────────────────
    # Turns 1 query into 4 alternative phrasings to widen the search surface.
    update("🔍 Query Expander: generating alternative search queries...")
    queries = _query_expander.expand(query)

    # ── Agent 2: Search Agent ─────────────────────────────────────────────────
    # Searches Semantic Scholar for each query, combines and deduplicates results.
    update(f"📚 Search Agent: searching across {len(queries)} queries...")
    papers = _search_agent.search(queries)

    # ── Agent 3: Relevance Ranker ─────────────────────────────────────────────
    # Scores each paper 1–10 for relevance, returns top 7.
    update(f"⚖️  Ranker: scoring {len(papers)} papers for relevance...")
    top7 = _ranker_agent.rank(query, papers)

    # ── Agent 4: Summarizer ───────────────────────────────────────────────────
    # Generates 3-sentence summaries for all 7 papers in parallel.
    update("✍️  Summarizer: generating summaries in parallel...")
    top7 = _summarizer_agent.summarize(top7)

    # ── Agent 5: Report Compiler ──────────────────────────────────────────────
    # Formats everything into a markdown report and a JSON-serializable dict.
    update("📝 Compiler: building final report...")
    markdown, report_dict = _report_compiler.compile(query, top7)

    update("✅ Done!")
    return markdown, report_dict


# ── Quick test ────────────────────────────────────────────────────────────────
# Run this file directly to test the full pipeline:
#   python pipeline.py
if __name__ == "__main__":
    import json

    test_query = "Transformers and self attention"
    print(f"Running pipeline for: {test_query!r}\n")

    markdown, report_dict = run_pipeline(
        test_query,
        status_callback=lambda msg: print(f"  {msg}")
    )

    print("\n" + "=" * 60)
    print(markdown[:1000])
    print("=" * 60)
    print(f"\nDict has {len(report_dict['papers'])} papers.")
    print("Pipeline complete.")
