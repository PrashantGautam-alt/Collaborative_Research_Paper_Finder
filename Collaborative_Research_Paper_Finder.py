import streamlit as st
import arxiv
import pandas as pd
import time
from openai import OpenAI

# --- CONFIGURATION ---
# Replace with your API Key or use st.secrets in production
# st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE") 

st.set_page_config(page_title="A2A Research Finder", layout="wide")

# --- SHARED STATE ---
if "logs" not in st.session_state:
    st.session_state.logs = []
if "results" not in st.session_state:
    st.session_state.results = None

# --- UTILS ---
def log_message(sender, receiver, message):
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"**[{timestamp}] {sender} → {receiver}:** {message}"
    st.session_state.logs.append(log_entry)
    # Force sidebar update
    with st.sidebar:
        st.markdown(log_entry)
        st.divider()

def query_llm(system_prompt, user_content):
    """Simple wrapper for LLM calls"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Use a fast, cheap model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to LLM: {e}"

# --- AGENT CLASSES ---

class SearchAgent:
    def find_papers(self, topic):
        log_message("System", "Search Agent", f"Initiating search for: '{topic}'")
        
        # Search ArXiv
        search = arxiv.Search(
            query=topic,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            papers.append({
                "title": result.title,
                "summary": result.summary,
                "url": result.pdf_url,
                "authors": ", ".join([a.name for a in result.authors]),
                "published": result.published.strftime("%Y-%m-%d")
            })
            
        log_message("Search Agent", "Filter Agent", f"Found {len(papers)} raw papers. Handing over for filtering.")
        return papers

class FilterAgent:
    def filter_papers(self, papers, topic):
        log_message("Filter Agent", "Internal", "Analyzing relevance of papers...")
        
        # We will batch the titles to save tokens
        titles = "\n".join([f"{i}. {p['title']}" for i, p in enumerate(papers)])
        
        prompt = f"""
        You are a strict research filter. 
        Topic: {topic}
        
        Below is a list of papers. Return ONLY the indices (0-9) of the top 3-5 papers that are strictly relevant to the topic.
        Format: 0, 2, 5
        
        Papers:
        {titles}
        """
        
        response = query_llm("You are a helpful research assistant.", prompt)
        
        try:
            indices = [int(i.strip()) for i in response.split(',')]
            filtered_papers = [papers[i] for i in indices if i < len(papers)]
        except:
            # Fallback if LLM fails formatting
            filtered_papers = papers[:3]

        log_message("Filter Agent", "Summary Agent", f"Selected {len(filtered_papers)} high-relevance papers. Proceed to reading.")
        return filtered_papers

class SummaryAgent:
    def summarize(self, papers):
        log_message("Summary Agent", "Internal", "Reading abstracts and extracting key methodologies...")
        progress_bar = st.progress(0)
        
        enriched_papers = []
        
        for idx, p in enumerate(papers):
            prompt = f"""
            Analyze this abstract:
            {p['summary']}
            
            Extract 3 things strictly in this format:
            1. Methodology: (1 sentence)
            2. Key Result: (1 sentence)
            3. Limitation: (1 sentence)
            """
            analysis = query_llm("You are a scientific summarizer.", prompt)
            p['analysis'] = analysis
            enriched_papers.append(p)
            progress_bar.progress((idx + 1) / len(papers))
            
        log_message("Summary Agent", "Comparison Agent", "Analysis complete. Data ready for synthesis.")
        progress_bar.empty()
        return enriched_papers

class ComparisonAgent:
    def compare(self, papers, topic):
        log_message("Comparison Agent", "Internal", "Synthesizing findings and looking for patterns...")
        
        context = "\n\n".join([f"Paper: {p['title']}\nAnalysis: {p['analysis']}" for p in papers])
        
        prompt = f"""
        Based on these paper summaries regarding '{topic}', write a brief 'State of the Art' synthesis.
        Compare the methodologies used (e.g., Paper A used X, while Paper B used Y).
        """
        
        synthesis = query_llm("You are a Lead Researcher.", prompt)
        
        log_message("Comparison Agent", "Presentation Agent", "Synthesis complete. Generating final report.")
        return synthesis, papers

# --- UI LAYOUT ---

st.title("📚 A2A Collaborative Research Finder")
st.markdown("### Multi-Agent System: Search → Filter → Summary → Compare")

# Sidebar for A2A Logs
with st.sidebar:
    st.header("🤖 Agent Communication Logs")
    st.info("Watch the agents talk to each other here.")
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()
    
    # Render existing logs
    for log in st.session_state.logs:
        st.markdown(log)
        st.divider()

# Main Input
query = st.text_input("Research Topic", "Transformers vs LSTM for Time Series")
start_btn = st.button("Start Research Agents")

if start_btn and query:
    # Reset logs for new run
    st.session_state.logs = []
    
    # Instantiate Agents
    searcher = SearchAgent()
    filterer = FilterAgent()
    summarizer = SummaryAgent()
    comparer = ComparisonAgent()
    
    with st.spinner("Agents are collaborating..."):
        # 1. Search
        raw_papers = searcher.find_papers(query)
        
        # 2. Filter
        relevant_papers = filterer.filter_papers(raw_papers, query)
        
        # 3. Summarize
        analyzed_papers = summarizer.summarize(relevant_papers)
        
        # 4. Compare
        synthesis, final_data = comparer.compare(analyzed_papers, query)
        
        st.session_state.results = {
            "synthesis": synthesis,
            "data": final_data
        }

# --- PRESENTATION AGENT (The UI Renderer) ---
if st.session_state.results:
    res = st.session_state.results
    
    st.divider()
    st.subheader("📝 Comparative Synthesis")
    st.write(res['synthesis'])
    
    st.divider()
    st.subheader("📄 Top Selected Papers")
    
    # Create a nice layout for cards
    cols = st.columns(len(res['data']))
    
    for idx, paper in enumerate(res['data']):
        with st.expander(f"📌 {paper['title']}"):
            st.markdown(f"**Published:** {paper['published']}")
            st.markdown(f"**Authors:** {paper['authors']}")
            st.info(paper['analysis'])
            st.markdown(f"[Read PDF]({paper['url']})")

    # Data Table View
    st.divider()
    st.subheader("📊 Comparison Table")
    df = pd.DataFrame(res['data'])
    st.dataframe(df[['title', 'published', 'analysis']])