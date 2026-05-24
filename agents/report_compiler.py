from datetime import datetime


class ReportCompiler:
    """
    Agent 5 — Report Compiler

    Why this agent exists:
        The previous agents produce data — dicts with scores, summaries, metadata.
        This agent's only job is to format that data into something a human can read.
        No LLM calls. No API. Pure Python string formatting.

    Input:  original query string, list of 7 paper dicts
            (each paper has: title, authors, year, citation_count, url,
             relevance_score, relevance_reason, summary)
    Output: tuple of (markdown_string, report_dict)
            markdown_string — formatted report ready to display in Streamlit
            report_dict     — same data as a Python dict, serializable to JSON
    """

    def compile(self, query: str, papers: list[dict]) -> tuple[str, dict]:
        """
        Format the ranked, summarized papers into a markdown report and a dict.

        Args:
            query  — the user's original search topic
            papers — list of 7 paper dicts from SummarizerAgent

        Returns:
            (markdown_string, report_dict)
        """

        markdown = self._build_markdown(query, papers)
        report_dict = self._build_dict(query, papers)
        return markdown, report_dict

    def _build_markdown(self, query: str, papers: list[dict]) -> str:
        """
        Build a formatted markdown string from the paper list.

        Uses an f-string for the header and a loop for the individual papers.
        String concatenation (+=) appends each paper's section to the report.
        """

        top_paper = papers[0] if papers else {}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Header section
        report = f"""# Research Report: {query}
**Generated:** {timestamp}

---

## Summary
**{len(papers)} papers analyzed.**
Top result: *{top_paper.get('title', 'N/A')}*
({top_paper.get('year', 'N/A')} · {top_paper.get('citation_count', 0):,} citations)

---

## Papers Ranked by Relevance
"""

        # One section per paper
        for i, paper in enumerate(papers, start=1):
            # authors is a list of strings — join them with ", "
            authors_str = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors_str += " et al."

            report += f"""
### {i}. {paper.get('title', 'Untitled')}
**Authors:** {authors_str} | **Year:** {paper.get('year', 'N/A')} | **Citations:** {paper.get('citation_count', 0):,}
**Relevance Score:** {paper.get('relevance_score', 0)}/10
**Why relevant:** {paper.get('relevance_reason', '')}
**Summary:** {paper.get('summary', '')}
**Link:** {paper.get('url', '')}

---
"""

        return report

    def _build_dict(self, query: str, papers: list[dict]) -> dict:
        """
        Build a structured Python dict of the report — serializable to JSON.

        This is what the Streamlit download button will offer as a .json file.
        Structured data is more useful than markdown for anyone who wants to
        process the results programmatically.
        """

        return {
            "query": query,
            "generated_at": datetime.now().isoformat(),
            "total_papers": len(papers),
            "papers": [
                {
                    "rank":             i + 1,
                    "title":            p.get("title", ""),
                    "authors":          p.get("authors", []),
                    "year":             p.get("year"),
                    "citation_count":   p.get("citation_count", 0),
                    "relevance_score":  p.get("relevance_score", 0),
                    "relevance_reason": p.get("relevance_reason", ""),
                    "summary":          p.get("summary", ""),
                    "url":              p.get("url", ""),
                }
                for i, p in enumerate(papers)
            ]
        }
