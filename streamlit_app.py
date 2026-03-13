"""
Streamlit frontend for the Autonomous Research Crew.

Run with:
    streamlit run streamlit_app.py
"""

import queue
import threading
import time

import streamlit as st
from research_crew import run_research

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Research Crew",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "running": False,
    "completed": [],   # list of {"agent": str, "output": str}
    "result": "",
    "error": "",
    "event_queue": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def _crew_worker(query: str, q: queue.Queue) -> None:
    """Run the research crew in a thread and post events to the queue."""
    try:
        def cb(event_type, data):
            q.put((event_type, data))

        result = run_research(query, event_callback=cb)
        q.put(("result", {"output": result}))
    except Exception as exc:
        q.put(("error", {"message": str(exc)}))
    finally:
        q.put(("done", {}))

# ---------------------------------------------------------------------------
# Agent step definitions
# ---------------------------------------------------------------------------
STEPS = [
    ("Research Planner",           "🗺️"),
    ("Research Paper Searcher",    "🔍"),
    ("Research Paper Summarizer",  "📝"),
    ("Research Reporter",          "📊"),
]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔬 Autonomous Research Crew")
st.caption("Powered by **CrewAI** · **HuggingFace Qwen2.5** · **arXiv**")
st.divider()

# ---------------------------------------------------------------------------
# Input & controls
# ---------------------------------------------------------------------------
query = st.text_input(
    "Research Topic",
    placeholder="e.g. Recent advancements in energy-efficient IoT systems",
    disabled=st.session_state.running,
    label_visibility="visible",
)

col_run, col_clear, _ = st.columns([1.2, 1, 5])

with col_run:
    run_clicked = st.button(
        "🚀 Run Research",
        disabled=st.session_state.running or not query.strip(),
        use_container_width=True,
        type="primary",
    )

with col_clear:
    clear_clicked = st.button(
        "🗑️  Clear",
        disabled=st.session_state.running,
        use_container_width=True,
    )

if clear_clicked:
    st.session_state.completed = []
    st.session_state.result = ""
    st.session_state.error = ""
    st.rerun()

# ---------------------------------------------------------------------------
# Start research
# ---------------------------------------------------------------------------
if run_clicked and query.strip():
    st.session_state.running = True
    st.session_state.completed = []
    st.session_state.result = ""
    st.session_state.error = ""

    q: queue.Queue = queue.Queue()
    st.session_state.event_queue = q

    threading.Thread(
        target=_crew_worker, args=(query.strip(), q), daemon=True
    ).start()

# ---------------------------------------------------------------------------
# Poll event queue on every rerun
# ---------------------------------------------------------------------------
if st.session_state.running and st.session_state.event_queue is not None:
    q = st.session_state.event_queue
    while not q.empty():
        event_type, data = q.get_nowait()
        if event_type == "task_complete":
            st.session_state.completed.append(data)
        elif event_type == "result":
            st.session_state.result = data["output"]
        elif event_type == "error":
            st.session_state.error = data["message"]
            st.session_state.running = False
        elif event_type == "done":
            st.session_state.running = False

# ---------------------------------------------------------------------------
# Progress cards
# ---------------------------------------------------------------------------
if (
    st.session_state.running
    or st.session_state.completed
    or st.session_state.result
    or st.session_state.error
):
    st.subheader("Agent Progress")

    completed_names = {t["agent"] for t in st.session_state.completed}
    active_idx = len(st.session_state.completed)

    for i, (name, icon) in enumerate(STEPS):
        if name in completed_names:
            badge = "✅ Done"
        elif i == active_idx and st.session_state.running:
            badge = "⏳ Running…"
        else:
            badge = "🕐 Waiting"

        label = f"{icon} **{name}** — {badge}"
        task_output = next(
            (t["output"] for t in st.session_state.completed if t["agent"] == name),
            None,
        )

        with st.expander(f"{icon} {name}  —  {badge}", expanded=False):
            if task_output:
                preview = task_output[:2000]
                if len(task_output) > 2000:
                    preview += "\n\n*(output truncated — full content in downloaded report)*"
                st.text(preview)
            else:
                st.caption("No output yet.")

    # Running spinner
    if st.session_state.running:
        st.info("⏳ Research in progress… this may take several minutes.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Auto-refresh while running (1-second polling loop)
# ---------------------------------------------------------------------------
if st.session_state.running:
    time.sleep(1)
    st.rerun()

# ---------------------------------------------------------------------------
# Error banner
# ---------------------------------------------------------------------------
if st.session_state.error:
    st.error(f"❌ {st.session_state.error}")

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
if st.session_state.result:
    st.divider()
    st.subheader("📄 Research Report")
    st.markdown(st.session_state.result)

    st.download_button(
        label="⬇️  Download Report (.md)",
        data=st.session_state.result,
        file_name="research_report.md",
        mime="text/markdown",
        type="primary",
    )
