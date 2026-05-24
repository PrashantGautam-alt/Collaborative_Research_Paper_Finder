from utils.semantic_scholar import search_semantic_scholar


class SearchAgent:
    """
    Agent 2 — Search Agent

    Why this agent exists:
        The Query Expander gives us 4 query strings (original + 3 alternatives).
        Each string captures a different angle of the user's topic.
        This agent searches Semantic Scholar for each one, then combines and
        deduplicates the results into a single clean list.

        Without deduplication, the same paper could appear up to 4 times —
        once per query — and the downstream Ranker would score it multiple times.

    Input:  list of query strings (normally 4, but works with any number)
    Output: deduplicated list of paper dicts (up to 32 in, fewer out after dedup)

    Each paper dict has these keys (guaranteed by search_semantic_scholar):
        paper_id, title, abstract, authors, year, citation_count, url
    """

    def search(self, queries: list[str]) -> list[dict]:
        """
        Search Semantic Scholar for each query and return deduplicated results.

        Args:
            queries — list of query strings from QueryExpander.expand()

        Returns:
            A deduplicated list of paper dicts. The dict keys come from
            search_semantic_scholar: paper_id, title, abstract, authors,
            year, citation_count, url.
        """

        # Step 1: collect every paper from every query into one flat list.
        # all_papers may contain duplicates at this point — that's expected.
        all_papers = []

        for query in queries:
            # limit=8 means we ask for 8 papers per query.
            # With 4 queries that's up to 32 papers before deduplication.
            # If a query returns [] (API error or no results), extend() just
            # adds nothing — the loop keeps going with the remaining queries.
            results = search_semantic_scholar(query, limit=8)
            all_papers.extend(results)

        # Step 2: deduplicate by paper_id.
        #
        # We build a dict where the key is paper_id and the value is the
        # full paper dict. If the same paper_id appears more than once
        # (same paper found by two different queries), the second assignment
        # just overwrites the first — same data, no information lost.
        #
        # Why paper_id and not title?
        # The same paper can appear with slightly different title strings
        # from different search results (e.g. capitalisation, punctuation).
        # paper_id is Semantic Scholar's stable unique identifier — it never
        # varies for the same paper regardless of which query found it.
        deduped: dict[str, dict] = {}

        for paper in all_papers:
            deduped[paper["paper_id"]] = paper

        # Step 3: return only the unique papers as a list.
        # .values() gives us a dict_values view of the paper dicts;
        # list() converts it to a plain list for the next agent.
        return list(deduped.values())
