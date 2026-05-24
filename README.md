# Collaborative Research Paper Finder

A multi-agent AI system that automates academic literature search. Given a research topic, five specialized agents work in sequence to expand the query, search Semantic Scholar, rank results by relevance, summarize each paper, and compile a structured report — all displayed through a Streamlit interface.

Live URL: https://prashantgautam-research-finder.streamlit.app/

---

## How It Works

The system runs a sequential pipeline of five agents:

1. **Query Expander** — Uses Gemini to generate three alternative phrasings of the user's topic, widening the search surface.
2. **Search Agent** — Queries Semantic Scholar for each phrasing and deduplicates the combined results by paper ID.
3. **Ranker Agent** — Scores each paper 1–10 for relevance to the original query and returns the top 7.
4. **Summarizer Agent** — Generates a structured 3-sentence summary for each paper in parallel using `asyncio`.
5. **Report Compiler** — Formats the results into a markdown report and a downloadable JSON file.

---

## Tech Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| LLM | Google Gemini (`gemini-2.5-flash-lite`) |
| Search API | Semantic Scholar Graph API |
| Language | Python 3.8+ |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/collaborative-research-paper-finder.git
cd collaborative-research-paper-finder
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

API keys are loaded from a `.env` file. A template is provided.

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
GEMINI_API_KEY=your_gemini_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

- **Gemini API key** — free at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- **Semantic Scholar API key** — free at [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) (optional, but increases rate limit from 1 to 100 requests/sec)

The `.env` file is listed in `.gitignore` and will not be committed.

---

## Usage

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

1. Enter a research topic.
2. Click **Find Papers**.
3. Wait for the pipeline to complete — each agent step is shown live.
4. Review the ranked paper summaries.
5. Download the report as a JSON file if needed.

---

## Project Structure

```
collaborative-research-paper-finder/
│
├── app.py                    # Streamlit UI — entry point
├── pipeline.py               # Orchestrator — runs all 5 agents in sequence
│
├── agents/
│   ├── query_expander.py     # Agent 1: expands the user query into 4 phrasings
│   ├── search_agent.py       # Agent 2: searches Semantic Scholar, deduplicates
│   ├── ranker_agent.py       # Agent 3: scores papers for relevance, returns top 7
│   ├── summarizer_agent.py   # Agent 4: generates parallel 3-sentence summaries
│   └── report_compiler.py   # Agent 5: formats results into markdown and JSON
│
├── utils/
│   ├── gemini_client.py      # Shared Gemini client used by all LLM agents
│   └── semantic_scholar.py   # Semantic Scholar API wrapper
│
├── archive/                  # Original single-file prototype (reference only)
│
├── .env.example              # API key template — copy to .env and fill in keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Disclaimer

This project uses the Semantic Scholar API but is not affiliated with or endorsed by Semantic Scholar or ArXiv. LLMs may produce inaccurate summaries — always verify claims against the original papers.
