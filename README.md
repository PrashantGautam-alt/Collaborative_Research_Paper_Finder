# Collaborative Research Paper Finder

Tired of manually searching across three different databases for a literature review,
so I built a pipeline where five specialized agents do it for you.
Give it a topic, get back ranked papers with structured summaries and a downloadable report.

Live: https://prashantgautam-research-finder.streamlit.app/

---

## How It Works

Sequential pipeline of five agents:

1. **Query Expander** — Gemini generates three alternative phrasings of your topic, widening the search surface.
2. **Search Agent** — Queries Semantic Scholar for each phrasing, deduplicates by paper ID.
3. **Ranker Agent** — Scores each paper 1-10 for relevance, returns the top 7.
4. **Summarizer Agent** — Generates a 3-sentence structured summary per paper, run in parallel with `asyncio`.
5. **Report Compiler** — Formats everything into a markdown report and a downloadable JSON.

---

## Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| LLM | Google Gemini (gemini-2.5-flash-lite) |
| Search | Semantic Scholar Graph API |
| Language | Python 3.8+ |

---

## Setup

```bash
git clone https://github.com/your-username/collaborative-research-paper-finder.git
cd collaborative-research-paper-finder
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
streamlit run app.py

You need two API keys in .env:
- GEMINI_API_KEY — free at aistudio.google.com
- SEMANTIC_SCHOLAR_API_KEY — free at semanticscholar.org/product/api
(optional, but bumps the rate limit from 1 to 100 req/sec)

---
Project Structure

├── app.py                    # Streamlit UI, entry point
├── pipeline.py               # runs all 5 agents in sequence
├── agents/
│   ├── query_expander.py
│   ├── search_agent.py
│   ├── ranker_agent.py
│   ├── summarizer_agent.py
│   └── report_compiler.py
└── utils/
    ├── gemini_client.py      # shared Gemini client
    └── semantic_scholar.py   # API wrapper

---
Note: LLM summaries may contain inaccuracies. Always verify against the original papers.
