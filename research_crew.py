import os
import time
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, LLM, Process
from tools import search_arxiv_tool

load_dotenv()  # loads HUGGINGFACEHUB_API_TOKEN from .env


AGENTS_ORDER = [
    "Research Planner",
    "Research Paper Searcher",
    "Research Paper Summarizer",
    "Research Reporter",
]


def _make_llm(max_tokens: int = 400, temperature: float = 0.3):
    """HuggingFace Inference API — free tier caps each response at ~500 tokens."""
    return LLM(
        model="huggingface/Qwen/Qwen2.5-72B-Instruct",
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _make_task_callback(agent_name: str, event_callback):
    """Return a crewAI task callback that fires an SSE event."""
    def cb(task_output):
        if event_callback:
            event_callback("task_complete", {
                "agent": agent_name,
                "output": str(task_output),
            })
    return cb


def run_research(query: str, event_callback=None):
    """
    Build and run the research crew for *query*.

    event_callback(event_type: str, data: dict | str) is called for:
      - ("agent_start",  {"agent": name})
      - ("task_complete", {"agent": name, "output": str})
      - ("result",        str)          <- final markdown report
      - ("error",         str)
    """
    verbose_mode = event_callback is None   # verbose in CLI, silent in API mode

    # HuggingFace free tier hard-caps each response at ~500 tokens — keep all budgets under that
    llm_light  = _make_llm(max_tokens=400)   # planner: just 3 query lines
    llm_search = _make_llm(max_tokens=600)   # searcher/summarizer: compact lists
    llm_report = _make_llm(max_tokens=700, temperature=0.4)  # reporter: one section at a time

    # --- Agents ---
    planner = Agent(
        role="Research Planner",
        goal=(
            "Decompose the research topic into exactly 3 focused, non-overlapping arXiv "
            "search queries that together cover the breadth of the subject."
        ),
        backstory=(
            "You are a seasoned research strategist who converts vague topics into "
            "precise, high-yield search strings. You output only the numbered list — "
            "no preamble, no explanation."
        ),
        llm=llm_light,
        verbose=verbose_mode,
        allow_delegation=False,
    )

    searcher = Agent(
        role="Research Paper Searcher",
        goal=(
            "Execute each query from the planner using the Search arXiv tool. "
            "Return title, publication date, PDF URL, and a one-sentence summary for every paper found."
        ),
        backstory=(
            "You are a meticulous academic librarian. You call the search tool once per query, "
            "collect all results, and return a clean numbered list. You never fabricate URLs."
        ),
        llm=llm_search,
        tools=[search_arxiv_tool],
        verbose=verbose_mode,
        max_retry_limit=3,
        max_iter=6,
        allow_delegation=False,
    )

    summarizer = Agent(
        role="Research Paper Summarizer",
        goal=(
            "For each paper in the search results, extract: (1) the core technical "
            "contribution, (2) the method or architecture used, and (3) reported performance "
            "gains or benchmarks. Output as a structured bullet list per paper."
        ),
        backstory=(
            "You are a machine-learning researcher who reads dense papers daily. "
            "You distill technical depth into crisp, information-dense bullets without "
            "losing numerical results or key terminology."
        ),
        llm=llm_search,
        verbose=verbose_mode,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Research Reporter",
        goal=(
            "Write clear, concise, accurate Markdown report sections. "
            "Be factual and direct. One well-written paragraph per point is enough."
        ),
        backstory=(
            "You are a technical writer who values clarity and accuracy over length. "
            "You write focused, information-dense sections that never pad content."
        ),
        llm=llm_report,
        verbose=verbose_mode,
        allow_delegation=False,
    )

    # --- Tasks ---
    planner_task = Task(
        description=(
            "The user wants to research: {query}.\n"
            "Produce exactly 3 specific, distinct arXiv search queries that together "
            "cover the main angles of this topic. Output only a numbered list."
        ),
        expected_output="Numbered list of 3 arXiv search queries, one per line, no extra text.",
        agent=planner,
        callback=_make_task_callback("Research Planner", event_callback),
    )

    searcher_task = Task(
        description=(
            "Using the 3 queries provided by the planner, call the Search arXiv tool "
            "once for each query. Compile all unique results into a single numbered list. "
            "For each paper include: title, publication date, PDF URL, one-sentence summary."
        ),
        expected_output="Numbered list of papers. Each entry: Title | Date | URL | One-sentence summary.",
        agent=searcher,
        context=[planner_task],
        callback=_make_task_callback("Research Paper Searcher", event_callback),
    )

    summarizer_task = Task(
        description=(
            "For every paper in the search results, write a structured summary block:\n"
            "  - **Contribution**: what problem it solves\n"
            "  - **Method**: technique, model, or architecture used\n"
            "  - **Results**: key metrics or benchmark numbers\n"
            "Be concise but preserve all quantitative details."
        ),
        expected_output="One structured summary block per paper with Contribution, Method, and Results bullets.",
        agent=summarizer,
        context=[searcher_task],
        callback=_make_task_callback("Research Paper Summarizer", event_callback),
    )

    # --- Part 1: Introduction + Key Papers ---
    report_part1_task = Task(
        description=(
            "Write a concise Markdown report section for '{query}'.\n"
            "## Introduction: 2 short paragraphs — what the topic is and why it matters.\n"
            "## Key Papers: one short paragraph per paper — bold title, year, main contribution."
        ),
        expected_output="## Introduction and ## Key Papers sections in Markdown. Clear and factual.",
        agent=reporter,
        context=[planner_task, searcher_task, summarizer_task],
        callback=_make_task_callback("Research Reporter", event_callback),
    )

    # --- Part 2: Technical Analysis, Comparison Table, Conclusion ---
    report_part2_task = Task(
        description=(
            "Continue the report on '{query}'.\n"
            "## Technical Analysis: 2 paragraphs comparing the methods and results across papers.\n"
            "## Comparison Table: Markdown table — Paper | Method | Key Result.\n"
            "## Conclusion: 1 paragraph with key takeaways and 2 future directions."
        ),
        expected_output="## Technical Analysis, ## Comparison Table, and ## Conclusion in Markdown.",
        agent=reporter,
        context=[report_part1_task],
        callback=_make_task_callback("Research Reporter", event_callback),
        output_file="final_research_report.md",
    )

    # --- Crew: single sequential run (no rate-limit sleep needed for HuggingFace) ---
    crew1 = Crew(
        agents=[planner, searcher, summarizer, reporter],
        tasks=[planner_task, searcher_task, summarizer_task, report_part1_task],
        process=Process.sequential,
        verbose=verbose_mode,
    )
    crew1.kickoff(inputs={"query": query})

    crew2 = Crew(
        agents=[reporter],
        tasks=[report_part2_task],
        process=Process.sequential,
        verbose=verbose_mode,
    )
    crew2.kickoff(inputs={"query": query})

    # crew.kickoff() only returns the last task — manually join both parts for the full report
    part1 = str(report_part1_task.output) if report_part1_task.output else ""
    part2 = str(report_part2_task.output) if report_part2_task.output else ""
    full_report = f"# Research Report: {query}\n\n{part1}\n\n{part2}".strip()
    return full_report


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Recent advancements in energy-efficient IoT systems"
    result = run_research(query)
    print("\n\n" + "=" * 30)
    print("FINAL RESEARCH REPORT")
    print("=" * 30 + "\n")
    print(result)