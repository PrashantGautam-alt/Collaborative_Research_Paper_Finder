"""
Smoke test for the full 5-agent pipeline.

Both external services are mocked, so this test makes NO network calls and needs
NO real API key:
  - Semantic Scholar  -> agents.search_agent.search_semantic_scholar is patched
  - Gemini            -> each agent's shared `gemini_client` is replaced with a
                         MagicMock whose generate_content returns canned text

The fake Gemini response is chosen by inspecting the prompt, so the query
expander gets a JSON array, the ranker gets a JSON object, and the summarizer
gets plain text — exactly what each agent parses.

Run with:  pytest
"""

import os

# A dummy key must exist before `utils.gemini_client` is imported, because it
# constructs genai.Client(api_key=...) at import time. The value is never used —
# every Gemini call is mocked below.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

from unittest.mock import MagicMock, patch

import pipeline


class _Resp:
    """Minimal stand-in for a Gemini response object (only `.text` is used)."""
    def __init__(self, text: str):
        self.text = text


def _fake_generate_content(model=None, contents="", **kwargs):
    """Return a canned response shaped to match whichever agent is calling."""
    prompt = contents.lower()
    if "query expander" in prompt:
        return _Resp('["alternative one", "alternative two", "alternative three"]')
    if "relevance scorer" in prompt:
        return _Resp('{"score": 8, "reason": "directly addresses the query"}')
    # summarizer
    return _Resp("Solves a problem. Uses a method. Achieves a result.")


def _fake_search(query, limit=8):
    """Return 10 unique papers regardless of query (dedup leaves 10 -> top 7)."""
    return [
        {
            "paper_id": f"p{i}",
            "title": f"Paper {i}",
            "abstract": f"Abstract for paper {i}.",
            "authors": ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Edsger Dijkstra"],
            "year": 2020 + (i % 5),
            "citation_count": 100 * i,
            "url": f"https://example.org/p{i}",
        }
        for i in range(10)
    ]


def _fake_gemini_client():
    client = MagicMock()
    client.models.generate_content.side_effect = _fake_generate_content
    return client


def test_run_pipeline_end_to_end():
    fake_client = _fake_gemini_client()

    with patch("agents.search_agent.search_semantic_scholar", side_effect=_fake_search), \
         patch("agents.query_expander.gemini_client", fake_client), \
         patch("agents.ranker_agent.gemini_client", fake_client), \
         patch("agents.summarizer_agent.gemini_client", fake_client):

        markdown, report = pipeline.run_pipeline("attention mechanisms")

    # Markdown report
    assert isinstance(markdown, str)
    assert "Research Report" in markdown

    # Structured report
    assert isinstance(report, dict)
    assert report["query"] == "attention mechanisms"
    assert report["total_papers"] == 7          # ranker caps at top 7
    assert len(report["papers"]) == 7

    # Every paper is fully populated by the pipeline
    for p in report["papers"]:
        assert 1 <= p["relevance_score"] <= 10
        assert p["summary"]                       # non-empty summary
        assert p["url"].startswith("https://")

    # Ranks are 1..7 in order
    assert [p["rank"] for p in report["papers"]] == list(range(1, 8))
