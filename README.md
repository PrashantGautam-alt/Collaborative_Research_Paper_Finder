# Collaborative_Research_Paper_Finder


## Overview
The **Collaborative Research Paper Finder** is a multi-agent system (MAS) designed to automate the initial stages of academic literature review. By leveraging the **ArXiv API** and **Large Language Models (LLMs)**, the system orchestrates a team of specialized autonomous agents to search for, filter, analyze, and synthesize research papers.

The core innovation of this project is the **Agent-to-Agent (A2A) communication architecture**. Instead of relying on a single monolithic prompt, tasks are distributed across distinct agents that operate in a **sequential state pipeline**, passing structured data and reasoning logs between each other before presenting the final output to the user.

---

## System Architecture

The application implements a **Sequential State Pipeline** composed of five specialized agents:

1. **Search Agent**  
   Interfaces with the ArXiv API to retrieve raw paper metadata based on user queries.

2. **Filter Agent**  
   Acts as a gatekeeper by analyzing titles and abstracts to select only the most semantically relevant papers, reducing computational overhead for downstream agents.

3. **Summary Agent**  
   Processes selected abstracts to extract structured research components:
   - Methodology  
   - Key Results  
   - Limitations  

4. **Comparison Agent**  
   Aggregates individual paper analyses to generate a *State-of-the-Art* synthesis, identifying trends, patterns, and methodological differences across papers.

5. **Presentation Agent**  
   Renders the user interface, displaying:
   - Inter-agent communication logs (A2A logs)
   - Final comparative summaries and structured paper cards

---

## Features

- **Autonomous Literature Search**  
  Automated querying of the ArXiv database.

- **Intelligent Filtering**  
  LLM-based relevance scoring to eliminate low-quality or irrelevant results.

- **Structured Summarization**  
  Extraction of specific research components rather than generic summaries.

- **Comparative Synthesis**  
  Side-by-side comparison and high-level synthesis of multiple papers.

- **Transparent Logic**  
  Real-time display of agent message passing, improving explainability and trust.

---

## Technical Stack

- **Language:** Python 3.8+  
- **Interface:** Streamlit  
- **LLM Integration:** OpenAI API (GPT-4o-mini or compatible)  
- **Data Source:** ArXiv API  
- **Data Manipulation:** Pandas  

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- A valid OpenAI API key

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/a2a-research-finder.git
cd a2a-research-finder
````

---

### Step 2: Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate the environment:**

* **Windows**

```bash
venv\Scripts\activate
```

* **macOS / Linux**

```bash
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install streamlit openai arxiv pandas
```

---

### Step 4: Configuration

You must configure your OpenAI API key.

#### Option A: Environment Variable (Recommended)

```bash
export OPENAI_API_KEY="sk-..."
```

#### Option B: Direct Configuration

Open `app.py` and modify the client initialization:

```python
client = OpenAI(api_key="sk-...")
```

**Warning:** Do not commit API keys to public repositories.

---

## Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will launch in your browser at:

```
http://localhost:8501
```

### Workflow

1. Enter a research topic (e.g., *"Deep Learning for Time Series Forecasting"*).
2. Click **Start Research Agents**.
3. Observe agent communication logs in the sidebar.
4. Review the structured paper summaries and comparative synthesis.

---

## Project Structure

```plaintext
a2a-research-finder/
├── Collaborative_Research_Paper_Finder.py              # Main application logic and agent implementations
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Future Roadmap

* **Full PDF Parsing**
  Integration of PyMuPDF to analyze complete paper texts beyond abstracts.

* **Cyclic Agent Negotiation**
  Use of LangGraph to allow the Comparison Agent to trigger additional searches if data is insufficient.

* **Vector Storage & Retrieval**
  Integration of FAISS or Chroma for semantic search and Q&A over retrieved papers.



## Disclaimer

This tool uses the ArXiv API but is **not endorsed or certified by ArXiv**.
LLMs may hallucinate—always verify important citations and claims against the original research papers.

