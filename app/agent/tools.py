import logging
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

WebSearch = Callable[[str], list[dict]]


def make_serper_search(api_key: str, num_results: int = 4) -> WebSearch:
    """Build a web-search callable backed by Serper (Google results)."""

    def search(query: str) -> list[dict]:
        try:
            resp = httpx.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": num_results},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # network / quota / bad key: degrade gracefully
            logger.warning("web search failed: %s", exc)
            return []
        return [
            {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")}
            for r in data.get("organic", [])[:num_results]
        ]

    return search
