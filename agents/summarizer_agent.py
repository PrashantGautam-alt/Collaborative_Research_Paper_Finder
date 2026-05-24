import asyncio

from utils.gemini_client import gemini_client, GEMINI_MODEL


class SummarizerAgent:
    """
    Agent 4 — Summarizer

    Why this agent exists:
        The Ranker gives us 7 relevant papers but the abstracts are dense and
        academic. This agent generates a plain 3-sentence summary for each paper:
        what problem it solves, what approach it uses, what result it achieves.

    Why parallel (asyncio.gather):
        Each Gemini call takes ~2–3 seconds. Sequentially, 7 summaries = ~21s.
        In parallel, all 7 fire at the same time, so total time ≈ ~3s.

    How parallelism works here:
        summarize()         — regular def, called by pipeline.py (synchronous world)
        _summarize_all()    — async def, orchestrates gather
        _summarize_one()    — async def, handles one paper using asyncio.to_thread

        asyncio.run() bridges the synchronous caller into the async world.

    Input:  list of 7 paper dicts (each must have 'title' and 'abstract')
    Output: same list with 'summary' field added to each paper
    """

    def summarize(self, papers: list[dict]) -> list[dict]:
        """
        Entry point — regular def so pipeline.py can call it without await.

        asyncio.run() creates a fresh event loop, runs _summarize_all() inside it,
        waits for it to finish, then returns the result back to the synchronous world.

        Args:
            papers — list of 7 paper dicts from RankerAgent

        Returns:
            Same list with 'summary' field added to every paper.
        """
        return asyncio.run(self._summarize_all(papers))

    async def _summarize_all(self, papers: list[dict]) -> list[dict]:
        """
        Async orchestrator — creates all tasks and runs them in parallel.

        Steps:
            1. Build a list of coroutines (one per paper) — not started yet
            2. asyncio.gather(*tasks) starts ALL of them at the same time
            3. Wait until every single one finishes
            4. Results come back in the same order as the input papers
        """

        # List comprehension creates 7 coroutines — one _summarize_one call per paper.
        # At this point nothing has run yet — these are just "tasks waiting to start".
        tasks = [self._summarize_one(paper) for paper in papers]

        # asyncio.gather(*tasks) unpacks the list and launches all 7 simultaneously.
        # await here means: pause until ALL 7 are done, then continue.
        # results is a list of paper dicts in the same order as `papers`.
        results = await asyncio.gather(*tasks)

        return list(results)

    async def _summarize_one(self, paper: dict) -> dict:
        """
        Async worker — summarizes one paper using Gemini.

        Why asyncio.to_thread:
            gemini_client.models.generate_content() is a blocking synchronous
            function. If called directly inside async code, it freezes the entire
            event loop — nothing else can run while it waits.
            asyncio.to_thread() runs it in a background thread instead, so the
            event loop stays free to handle the other 6 papers simultaneously.

        Args:
            paper — one paper dict (needs 'title' and 'abstract')

        Returns:
            Same paper dict with 'summary' field added.
            summary = "" if Gemini fails for any reason.
        """

        prompt = f"""Summarize this research paper in exactly 3 sentences.
Sentence 1: What problem does this paper solve?
Sentence 2: What approach or method does it use?
Sentence 3: What is the main result or finding?

Paper title: {paper['title']}
Abstract: {paper['abstract']}

Return only the 3-sentence summary. No labels, no bullet points."""

        try:
            # asyncio.to_thread runs the blocking Gemini call in a background thread.
            # The first argument is the function to call.
            # The remaining arguments are passed to that function.
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt
            )

            paper["summary"] = response.text.strip()
            return paper

        except Exception as e:
            # Any failure (network, quota, timeout) → empty summary, keep going.
            # The pipeline does not crash because of one bad Gemini response.
            print(f"[SummarizerAgent] Failed for {paper['title'][:60]!r}: {e}")
            paper["summary"] = ""
            return paper
