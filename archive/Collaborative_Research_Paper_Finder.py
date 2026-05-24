import streamlit as st
import arxiv
import pandas as pd
import time
import re
import random
from google import genai


# ── API Key ────────────────────────────────────────────────────────────────────
api_key = "YOUR_API_KEY"

# Connect to the Gemini AI service using your key
# We create one shared client so we are not reconnecting on every single request
gemini_client = genai.Client(api_key=api_key)

# This is the Gemini model we will use for all AI tasks
# gemini-2.0-flash is fast, free-tier friendly, and handles research tasks well
GEMINI_MODEL = "gemini-2.0-flash"


# ── Page Config ────────────────────────────────────────────────────────────────
# Set the page title and use a wide layout so the content has more room to breathe
st.set_page_config(page_title="A2A Research Finder", layout="wide")


# ── Session State ──────────────────────────────────────────────────────────────
# Streamlit reruns the whole script every time you click a button or type something
# session_state lets us save data between those reruns so we don't lose our results
if "logs" not in st.session_state:
    st.session_state.logs = []

if "results" not in st.session_state:
    st.session_state.results = None

# Cache arXiv results per query so the same search never hits the API twice
if "arxiv_cache" not in st.session_state:
    st.session_state.arxiv_cache = {}

# Cache full pipeline results per query so repeat searches cost zero Gemini calls
if "gemini_cache" not in st.session_state:
    st.session_state.gemini_cache = {}


# ── Helpers ────────────────────────────────────────────────────────────────────
def log_message(sender, receiver, message):
    """Build a timestamped message and save it to the sidebar log."""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"**[{timestamp}] {sender} → {receiver}:** {message}"
    st.session_state.logs.append(log_entry)
    with st.sidebar:
        st.markdown(log_entry)
        st.divider()


def query_llm(system_prompt, user_content, max_retries=3):
    """Send a question to Gemini and get back the answer as plain text.

    Retries with exponential backoff on quota / rate-limit errors (429).
    Raises a clean, readable exception on all other failures so the caller
    can show a proper error — never returns a raw exception string.
    """
    full_prompt = f"{system_prompt}\n\n{user_content}"

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt
            )
            return response.text

        except Exception as e:
            err_str = str(e)

            # Detect Gemini rate-limit / quota errors (HTTP 429 / RESOURCE_EXHAUSTED)
            is_quota_error = (
                "429" in err_str or
                "RESOURCE_EXHAUSTED" in err_str or
                "quota" in err_str.lower() or
                "rate" in err_str.lower()
            )

            if is_quota_error and attempt < max_retries - 1:
                wait = (2 ** attempt) * 15 + random.uniform(1, 5)  # 15s, 30s, …
                log_message(
                    "Gemini", "System",
                    f"⚠️ Gemini quota/rate-limit hit. "
                    f"Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries - 1}…"
                )
                time.sleep(wait)
            elif is_quota_error:
                # All retries exhausted — raise a clean, readable error
                raise RuntimeError(
                    "🚫 Gemini API quota exceeded.\n\n"
                    "Your free-tier daily or per-minute limit has been reached. "
                    "Options:\n"
                    "• Wait a few minutes and try again (per-minute limit)\n"
                    "• Wait until tomorrow (daily limit resets at midnight Pacific)\n"
                    "• Upgrade to a paid Gemini plan for higher limits\n"
                    "• Check usage at: https://ai.google.dev/gemini-api/docs/rate-limits"
                ) from None
            else:
                # Non-quota error — raise a clean one-liner, not the raw JSON blob
                raise RuntimeError(f"Gemini API error: {type(e).__name__}") from e


# ── Search Agent ───────────────────────────────────────────────────────────────
# This agent's only job is to search ArXiv and return a list of papers
class SearchAgent:
    def find_papers(self, topic, max_retries=4):
        log_message("System", "Search Agent", f"Starting search for: '{topic}'")

        # ── Cache check ────────────────────────────────────────────────────────
        # arXiv enforces strict rate limits (HTTP 429).  Re-using cached results
        # for the same query avoids hitting the API more than once per session.
        cache_key = topic.strip().lower()
        if cache_key in st.session_state.arxiv_cache:
            cached = st.session_state.arxiv_cache[cache_key]
            log_message("Search Agent", "Filter Agent",
                        f"Cache hit! Returning {len(cached)} previously fetched papers.")
            return cached

        # Fetch 15 papers so the filter agent has a big enough pool to find 5 good ones
        search = arxiv.Search(
            query=topic,
            max_results=15,
            sort_by=arxiv.SortCriterion.Relevance
        )

        # delay_seconds=10 gives arXiv enough breathing room between paginated
        # requests.  num_retries is kept low because WE handle retries ourselves
        # with proper exponential backoff below.
        arxiv_client = arxiv.Client(
            page_size=15,
            delay_seconds=10.0,   # was 3.0 — too aggressive for arXiv's rate limiter
            num_retries=1         # let our own retry loop handle 429s
        )

        papers = []
        for attempt in range(max_retries):
            try:
                papers = []
                for result in arxiv_client.results(search):
                    papers.append({
                        "title":     result.title,
                        "summary":   result.summary,
                        "url":       result.pdf_url,
                        "authors":   ", ".join([a.name for a in result.authors]),
                        "published": result.published.strftime("%Y-%m-%d")
                    })
                break  # ✅ Success — exit the retry loop

            except arxiv.HTTPError as e:
                if e.status == 429 and attempt < max_retries - 1:
                    # Exponential backoff: 10s, 20s, 40s … + small random jitter
                    # so parallel users don't all retry at exactly the same moment.
                    wait = (2 ** attempt) * 10 + random.uniform(1, 5)
                    log_message(
                        "Search Agent", "System",
                        f"⚠️ arXiv rate-limited us (HTTP 429). "
                        f"Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries - 1}…"
                    )
                    time.sleep(wait)
                else:
                    # Final attempt failed or it's a different HTTP error — re-raise
                    raise

        # ── Store in session cache ─────────────────────────────────────────────
        st.session_state.arxiv_cache[cache_key] = papers

        log_message("Search Agent", "Filter Agent",
                    f"Found {len(papers)} papers. Sending them for filtering.")
        return papers


# ── Filter Agent ───────────────────────────────────────────────────────────────
# This agent asks the AI to pick only the most relevant papers
class FilterAgent:
    def filter_papers(self, papers, topic):
        log_message("Filter Agent", "Internal",
                    "Checking which papers actually match the topic...")

        # Only send titles to the AI — saves tokens and is fast enough for filtering
        titles = "\n".join([f"{i}. {p['title']}" for i, p in enumerate(papers)])

        prompt = f"""
You are a strict research filter.
Topic: {topic}

Below is a list of papers. Return ONLY the indices (numbers) of the top 3–5 papers
that are strictly relevant to the topic.
Format your answer as a comma-separated list of numbers, e.g.: 0, 2, 5

Papers:
{titles}
"""

        response = query_llm("You are a helpful research assistant.", prompt)

        # Pull all numbers out of the response — handles many possible reply formats
        indices = [int(n) for n in re.findall(r'\b\d+\b', response)
                   if int(n) < len(papers)]

        # Fallback: if the AI returned nothing useful, just take the first 3
        filtered_papers = [papers[i] for i in indices] if indices else papers[:3]

        log_message("Filter Agent", "Summary Agent",
                    f"Kept {len(filtered_papers)} relevant papers. Passing them on.")
        return filtered_papers


# ── Summary Agent ──────────────────────────────────────────────────────────────
# This agent reads all abstracts in ONE Gemini call and extracts key points
class SummaryAgent:
    def summarize(self, papers):
        log_message("Summary Agent", "Internal",
                    f"Sending all {len(papers)} abstracts in one batch call to Gemini…")

        # Build one combined prompt with all abstracts numbered
        # This is the key optimisation: 1 API call instead of N calls
        all_abstracts = "\n\n".join([
            f"Paper {i + 1} — {p['title']}:\n{p['summary']}"
            for i, p in enumerate(papers)
        ])

        prompt = f"""You are analyzing {len(papers)} research paper abstracts.

For EACH paper use EXACTLY this format, including the separator line:

---PAPER 1---
1. Methodology: (one sentence)
2. Key Result: (one sentence)
3. Limitation: (one sentence)

---PAPER 2---
1. Methodology: (one sentence)
2. Key Result: (one sentence)
3. Limitation: (one sentence)

(continue for all {len(papers)} papers)

Papers to analyze:

{all_abstracts}
"""
        response = query_llm("You are a scientific summarizer.", prompt)

        # Split the response back into one section per paper using the separator
        sections = re.split(r'---PAPER\s+\d+---', response)
        sections = [s.strip() for s in sections if s.strip()]

        for i, p in enumerate(papers):
            # Assign each section to its paper; fallback if Gemini skipped one
            p['analysis'] = sections[i] if i < len(sections) else "Analysis not available."

        log_message("Summary Agent", "Comparison Agent",
                    f"Done! Summarized {len(papers)} papers in 1 API call instead of {len(papers)}.")
        return papers


# ── Comparison Agent ───────────────────────────────────────────────────────────
# This agent writes a combined "State of the Art" summary across all papers
class ComparisonAgent:
    def compare(self, papers, topic):
        log_message("Comparison Agent", "Internal",
                    "Comparing all papers and finding common patterns...")

        context = "\n\n".join([
            f"Paper: {p['title']}\nAnalysis: {p['analysis']}"
            for p in papers
        ])

        prompt = f"""
Based on these paper summaries regarding '{topic}', write a brief 'State of the Art' synthesis.
Compare the methodologies used (e.g., Paper A used X, while Paper B used Y).

Papers:
{context}
"""

        synthesis = query_llm("You are a Lead Researcher.", prompt)
        log_message("Comparison Agent", "Presentation Agent",
                    "Comparison done. Final report is ready.")
        return synthesis, papers


# ── Page Layout ────────────────────────────────────────────────────────────────
st.title("A2A Collaborative Research Finder")
st.markdown("### Multi-Agent System: Search → Filter → Summarize → Compare")

# Sidebar: live agent communication log
with st.sidebar:
    st.header("Agent Communication Logs")
    st.info("Watch the agents talk to each other here in real time.")

    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()

    for log in st.session_state.logs:
        st.markdown(log)
        st.divider()


# Main input
query = st.text_input("Research Topic", "Transformers vs LSTM for Time Series")
start_btn = st.button("Start Research Agents")


# ── Run Pipeline ───────────────────────────────────────────────────────────────
if start_btn and query:
    st.session_state.logs = []   # clear logs from previous run
    gemini_key = query.strip().lower()

    # ── Gemini cache check ─────────────────────────────────────────────────────
    # If we already ran this exact query this session, skip ALL Gemini calls
    # and instantly show the saved results — zero API usage.
    if gemini_key in st.session_state.gemini_cache:
        log_message("System", "All Agents",
                    "✅ Cache hit! Using saved results — 0 Gemini API calls needed.")
        st.session_state.results = st.session_state.gemini_cache[gemini_key]

    else:
        searcher   = SearchAgent()
        filterer   = FilterAgent()
        summarizer = SummaryAgent()
        comparer   = ComparisonAgent()

        with st.spinner("Agents are working, this may take a minute…"):
            try:
                raw_papers      = searcher.find_papers(query)
                relevant_papers = filterer.filter_papers(raw_papers, query)
                analyzed_papers = summarizer.summarize(relevant_papers)
                synthesis, final_data = comparer.compare(analyzed_papers, query)

                results = {
                    "synthesis": synthesis,
                    "data":      final_data
                }
                st.session_state.results = results
                # Save to cache so the next run of this query costs nothing
                st.session_state.gemini_cache[gemini_key] = results

            except arxiv.HTTPError as e:
                if e.status == 429:
                    st.error(
                        "⏳ **arXiv rate limit hit (HTTP 429 — Too Many Requests).**\n\n"
                        "arXiv's public API allows only a few requests per minute. "
                        "Please wait **60–90 seconds** and try again. "
                        "Cached results will be used for the same query within this session."
                    )
                else:
                    st.error(f"❌ arXiv API error (HTTP {e.status}): {e}")

            except RuntimeError as e:
                err_msg = str(e)
                if "Gemini API quota exceeded" in err_msg:
                    st.error(err_msg)
                    st.info(
                        "💡 **Tip:** The free Gemini tier allows ~15 requests/minute. "
                        "Each search now uses only 3 Gemini calls (filter + batch summary + compare). "
                        "Wait a minute and try again, or try a different search phrase."
                    )
                else:
                    st.error(f"❌ {err_msg}")

            except Exception as e:
                st.error(f"❌ Unexpected error: {type(e).__name__}: {e}")


# ── Presentation ───────────────────────────────────────────────────────────────
if st.session_state.results:
    res = st.session_state.results

    st.divider()
    st.subheader("Comparative Synthesis")
    st.write(res["synthesis"])

    st.divider()
    st.subheader("Top Selected Papers")

    for paper in res["data"]:
        with st.expander(paper['title']):
            st.markdown(f"**Published:** {paper['published']}")
            st.markdown(f"**Authors:** {paper['authors']}")
            st.info(paper["analysis"])
            st.markdown(f"[Read the PDF]({paper['url']})")

    st.divider()
    st.subheader("Comparison Table")
    df = pd.DataFrame(res["data"])
    st.dataframe(df[["title", "published", "analysis"]])
