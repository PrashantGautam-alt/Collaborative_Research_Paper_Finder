import json
import re

from utils.gemini_client import gemini_client, GEMINI_MODEL


class RankerAgent:
    """
    Agent 3 — Relevance Ranker

    Why this agent exists:
        The Search Agent returns up to 32 papers that match the query keywords.
        But keyword match ≠ relevance. A paper titled "Transformers for Power Grids"
        matches "transformers" but is irrelevant to an NLP query.

        This agent scores every paper individually for true relevance to the
        original query, then returns only the top 7.

    Why individual scoring (not one big prompt):
        If we sent all 32 papers in one prompt, Gemini loses track of papers
        listed near the end — it's a known LLM limitation with long contexts.
        Scoring one paper at a time is more reliable and consistent.

    Input:  original query string, list of paper dicts (any length)
    Output: top 7 paper dicts sorted by relevance_score descending,
            each with two new fields added:
                relevance_score  — int 1–10 (0 if Gemini failed for that paper)
                relevance_reason — one sentence string ("" if Gemini failed)
    """

    def rank(self, query: str, papers: list[dict]) -> list[dict]:
        """
        Score each paper for relevance to the query and return the top 7.

        Args:
            query  — the user's original search topic
            papers — list of paper dicts from SearchAgent

        Returns:
            List of up to 7 paper dicts, sorted by relevance_score descending.
            Each dict has relevance_score and relevance_reason added.
        """

        # Step 1 — score every paper individually
        scored_papers = []

        for paper in papers:
            scored_paper = self._score_paper(query, paper)
            scored_papers.append(scored_paper)

        # Step 2 — sort all papers by relevance_score, highest first.
        # key=lambda p: p["relevance_score"] tells sorted() what value to compare.
        # reverse=True means descending order (10 → 9 → 8 ... → 0).
        sorted_papers = sorted(
            scored_papers,
            key=lambda p: p["relevance_score"],
            reverse=True
        )

        # Step 3 — return only the top 7.
        # Papers that failed JSON parsing got score=0 and naturally fall to the
        # bottom of the sorted list, so they are excluded here automatically.
        return sorted_papers[:7]

    def _score_paper(self, query: str, paper: dict) -> dict:
        """
        Ask Gemini to score one paper's relevance to the query.

        Adds relevance_score and relevance_reason to the paper dict.
        On any failure, sets relevance_score=0 and relevance_reason=""
        so the pipeline never crashes because of a single bad response.

        Args:
            query — the user's original search topic
            paper — one paper dict (must have 'title' and 'abstract' keys)

        Returns:
            The same paper dict with relevance_score and relevance_reason added.
        """

        # Build the prompt.
        # We give Gemini: the query, the paper title, and the abstract.
        # We ask for exactly one JSON object — no explanation, no extra text.
        # The example output format makes the expected structure explicit,
        # which reduces malformed responses.
        prompt = f"""You are a research paper relevance scorer.

Query: {query}
Paper title: {paper['title']}
Abstract: {paper['abstract']}

Score this paper's relevance to the query from 1 to 10.
Return JSON only. No explanation. No markdown fences.

Example output format:
{{"score": 7, "reason": "one sentence explaining relevance"}}"""

        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            response_text = response.text.strip()

            # Strip markdown code fences if Gemini wrapped the JSON in them.
            # Same pattern used in query_expander.py.
            # re.sub replaces all matches of the pattern with "" (empty string).
            # The pattern matches: ```json or ``` at the start/end of the block.
            cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response_text).strip()

            # json.loads() parses the cleaned string into a Python dict.
            # If Gemini returned anything other than valid JSON, this raises
            # json.JSONDecodeError and we jump to the except block below.
            result = json.loads(cleaned)

            # Safely read score and reason from the parsed dict.
            # .get() with a fallback means we don't crash if a key is missing.
            # int() converts the score in case Gemini returned it as a string.
            score = int(result.get("score", 0))
            reason = result.get("reason", "")

            # Clamp score to valid range 1–10.
            # If Gemini returns 11 or -1, this brings it back in range.
            score = max(1, min(10, score))

            # Add the two new fields to the paper dict and return it.
            paper["relevance_score"] = score
            paper["relevance_reason"] = reason
            return paper

        except (json.JSONDecodeError, ValueError, KeyError):
            # json.JSONDecodeError — Gemini didn't return valid JSON
            # ValueError          — int() failed (score wasn't a number)
            # KeyError            — unexpected dict structure
            # In all cases: assign score 0 so this paper sorts to the bottom.
            print(f"[RankerAgent] Failed to parse score for: {paper['title'][:60]!r}")
            paper["relevance_score"] = 0
            paper["relevance_reason"] = ""
            return paper

        except Exception as e:
            # Catch-all: network error, quota error, anything else.
            # Same outcome — score 0, pipeline keeps running.
            print(f"[RankerAgent] Unexpected error for {paper['title'][:60]!r}: {e}")
            paper["relevance_score"] = 0
            paper["relevance_reason"] = ""
            return paper
