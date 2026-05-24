import json
import streamlit as st

from pipeline import run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
# Sets the browser tab title and uses wide layout for more reading space.
st.set_page_config(
    page_title="Research Paper Finder",
    page_icon="🔬",
    layout="wide"
)

# ── Title and description ─────────────────────────────────────────────────────
st.title("🔬 Collaborative Research Paper Finder")
st.caption(
    "5-agent AI pipeline · Semantic Scholar · Gemini · "
    "Query Expansion → Search → Rank → Summarize → Report"
)
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
# st.text_input renders a text box and returns whatever the user typed.
# On first load this returns "" (empty string).
query = st.text_input(
    label="Enter your research topic",
    placeholder="e.g. attention mechanisms in deep learning",
)

# st.button returns True only on the single rerun triggered by the click.
# On every other rerun it returns False.
clicked = st.button("🔍 Find Papers", type="primary")

# ── Run the pipeline ──────────────────────────────────────────────────────────
if clicked:
    if not query.strip():
        # query.strip() removes whitespace — catches empty or blank input.
        st.error("Please enter a research topic before searching.")

    else:
        # st.status creates a collapsible progress panel.
        # As each agent runs, status.update() changes the label live in the browser.
        # state="complete" turns the panel green and collapses it when done.
        with st.status("Running pipeline...", expanded=True) as status:
            markdown, report_dict = run_pipeline(
                query.strip(),
                status_callback=lambda msg: status.update(label=msg)
            )
            status.update(label="✅ Done!", state="complete")

        # Store results in session_state so they survive the next rerun.
        # Without this, the results would vanish the moment Streamlit reruns.
        st.session_state["markdown"] = markdown
        st.session_state["report_dict"] = report_dict

# ── Display results ───────────────────────────────────────────────────────────
# This block runs on EVERY rerun — including the one after the pipeline finishes.
# As long as session_state has results, we show them.
if "markdown" in st.session_state:
    st.divider()

    # Download button — converts the dict to a JSON string for the file.
    # json.dumps() serializes the Python dict into a JSON-formatted string.
    # indent=2 makes the JSON file readable (not one long line).
    st.download_button(
        label="⬇️ Download Report (JSON)",
        data=json.dumps(st.session_state["report_dict"], indent=2),
        file_name="research_report.json",
        mime="application/json"
    )

    st.divider()

    # st.markdown renders the report string as formatted text —
    # headings, bold, links all render properly.
    st.markdown(st.session_state["markdown"])
