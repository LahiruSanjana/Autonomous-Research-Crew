import arxiv
from crewai.tools import tool

@tool("Search arXiv")
def search_arxiv_tool(query: str):
    """
    Searches arXiv for academic papers and returns concise titles and summaries.
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=5,  # broader coverage for better summarization
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for result in search.results():
            short_summary = (result.summary[:300] + '...') if len(result.summary) > 300 else result.summary
            results.append({
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors[:3]),  # first 3 authors
                "published": result.published.strftime('%Y-%m-%d'),
                "summary": short_summary,
                "url": result.pdf_url
            })
        return {"results": results} if results else {"results": "No papers found."}
    except Exception as e:
        return {"results": f"Error fetching papers: {e}"}