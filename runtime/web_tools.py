"""WEB_SEARCH and WEB_OPEN through EthosoftLib's web client and SearXNG.

Web access is opt-in: it is enabled only when a SearXNG URL is configured
(``pipeline.searxng_url`` or ``ETHOSOFT_SEARXNG_URL``).  Without one the
actions stay unconfigured instead of silently querying a public search engine.
WEB_OPEN keeps EthosoftLib's default of refusing loopback and private
addresses, since the URL comes from a model.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from runtime.local_tools import _observation


WEB_ACTIONS = frozenset({"WEB_SEARCH", "WEB_OPEN"})
SEARXNG_URL_ENV = "ETHOSOFT_SEARXNG_URL"

# Observations are serialized into every later Yasama step and into Yargi's
# prompt, so keep them well inside a 4k context.  Three results match the
# training data, whose web-search observations keep the source-ranked top 3
# (PROJECT_MASTER §6).
MAX_SEARCH_RESULTS = 3
MAX_PAGE_CHARS = 4000

ExecutorFn = Callable[[str, dict[str, Any]], Mapping[str, Any]]


def searxng_url(pipeline_cfg: Mapping[str, Any]) -> str | None:
    """Return the configured SearXNG URL; config wins over the environment."""
    url = str(pipeline_cfg.get("searxng_url") or os.environ.get(SEARXNG_URL_ENV) or "")
    return url.strip() or None


class WebExecutor:
    """Runs web actions itself and hands every other action to ``fallback``."""

    def __init__(self, base_url: str, fallback: ExecutorFn) -> None:
        try:
            from ethosoftlib.web import SearXNGSearch, WebClient, WebError
        except ImportError as exc:
            raise RuntimeError(
                "Web tools require EthosoftLib with the ethosoftlib.web module. "
                "Install with pip install -e '.[mercan]' or pip install -e /path/to/EthosoftLib."
            ) from exc
        self.base_url = base_url
        self._client = WebClient(provider=SearXNGSearch(base_url))
        self._web_error = WebError
        self._fallback = fallback

    def __call__(self, action: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
        if action not in WEB_ACTIONS:
            return self._fallback(action, arguments)
        try:
            if action == "WEB_SEARCH":
                query = str(arguments.get("query") or "")
                limit = min(int(arguments.get("max_results") or MAX_SEARCH_RESULTS), MAX_SEARCH_RESULTS)
                results = self._client.search(query, max_results=limit)
                return _observation(action, data={
                    "query": query,
                    "results": [result.to_dict() for result in results],
                })
            page = self._client.open_url(str(arguments.get("url") or ""), max_chars=MAX_PAGE_CHARS)
            return _observation(action, data=page.to_dict())
        except (self._web_error, ValueError) as exc:
            # A failed request is an observation for Yargi to report, not a crash.
            return _observation(action, error=str(exc))
