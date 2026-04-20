"""
Tool: web_search

Searches the web using DDGS (Dux Distributed Global Search),
a metasearch library that aggregates results from DuckDuckGo,
Bing, Google, and other search engines.

No API key required. This is what Loki uses when it says
"Let me look that up."

Install: pip install ddgs
"""

from typing import Any, Dict

from tool_registry import register_tool

# ─── Import check ──────────────────────────────────────────────────
# Primary: new 'ddgs' package (pip install ddgs)
# Fallback: old 'duckduckgo-search' package (deprecated, still works)
# If neither is installed, the tool returns a clean error message.

_DDGS_AVAILABLE = False
_USING_LEGACY = False
DDGS = None

try:
    from ddgs import DDGS  # type: ignore
    _DDGS_AVAILABLE = True
except ImportError:
    try:
        import warnings
        # Suppress all warnings from the deprecated package
        warnings.filterwarnings("ignore", module="duckduckgo_search")
        warnings.filterwarnings("ignore", message=".*has been renamed.*")
        from duckduckgo_search import DDGS  # type: ignore
        _DDGS_AVAILABLE = True
        _USING_LEGACY = True
    except ImportError:
        pass


@register_tool("web_search")
def web_search(query: str, num_results: int = 3) -> Dict[str, Any]:
    """Searches the web and returns top results with title, URL, and snippet.
    :param query: The search query string.
    :param num_results: Number of results to return (default 3, max 10).
    :return: A dict with the query and a list of result entries.
    """
    if not query or not query.strip():
        return {
            "error": "Empty search query"
        }

    if not _DDGS_AVAILABLE:
        return {
            "error": "ddgs is not installed. Run: pip install ddgs"
        }

    # Clamp num_results to a reasonable range
    num_results = max(1, min(num_results, 10))

    try:
        # Suppress deprecation warnings from legacy duckduckgo-search package
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ddgs = DDGS()
            raw_results = ddgs.text(query, max_results=num_results)

        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })

        return {
            "query": query,
            "num_results": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Search failed: {type(e).__name__}: {e}"
        }
