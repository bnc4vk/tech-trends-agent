from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlparse

import feedparser
import requests
from langchain_core.tools import tool

from .config import (
    BRAVE_SEARCH_API_KEY,
    REFERENCE_SEARCH_DEPTH,
    REFERENCE_SEARCH_MAX_RESULTS,
    SEARCH_MAX_QUERY_CHARS,
    SERPAPI_API_KEY,
    TAVILY_API_KEY,
)
from .schemas import SourceCandidate, SourceItem

DEFAULT_TIMEOUT = 20
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime(*value[:6])
    except Exception:
        return None


def _to_naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    if not value:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _to_naive_utc(parsed)
    except ValueError:
        return None


def _filter_recent(items: List[SourceItem], lookback_days: int) -> List[SourceItem]:
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    filtered: List[SourceItem] = []
    for item in items:
        if item.published_at:
            item.published_at = _to_naive_utc(item.published_at)
        if not item.published_at or item.published_at >= cutoff:
            filtered.append(item)
    return filtered


def _truncate_query(query: str) -> str:
    if len(query) <= SEARCH_MAX_QUERY_CHARS:
        return query
    truncated = query[:SEARCH_MAX_QUERY_CHARS]
    last_space = truncated.rfind(" ")
    if last_space > SEARCH_MAX_QUERY_CHARS * 0.8:
        return truncated[:last_space]
    return truncated


def _search_tavily(query: str, max_results: int, search_depth: str = "advanced") -> List[SourceCandidate]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not configured.")
    query = _truncate_query(query)
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": False,
        },
        timeout=DEFAULT_TIMEOUT,
        headers=DEFAULT_HEADERS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = f"{snippet[:500]}..."
        raise RuntimeError(f"Tavily error {response.status_code}: {snippet}") from exc
    payload = response.json()
    results: List[SourceCandidate] = []
    for item in payload.get("results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
            )
        )
    return results


def _search_brave_payload(query: str, max_results: int) -> dict:
    if not BRAVE_SEARCH_API_KEY:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not configured.")
    query = _truncate_query(query)
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        timeout=DEFAULT_TIMEOUT,
        headers={
            **DEFAULT_HEADERS,
            "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
        },
    )
    response.raise_for_status()
    return response.json()


def _search_serpapi_payload(query: str, max_results: int) -> dict:
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY is not configured.")
    query = _truncate_query(query)
    response = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": query, "num": max_results, "api_key": SERPAPI_API_KEY},
        timeout=DEFAULT_TIMEOUT,
        headers=DEFAULT_HEADERS,
    )
    response.raise_for_status()
    return response.json()


def _build_url_reference_query(source_url: str) -> str:
    domain = urlparse(source_url).netloc.lower()
    query = f"\"{source_url}\""
    if domain:
        query = f"{query} -site:{domain}"
    return _truncate_query(query)


def _build_title_reference_query(title: str, source_url: Optional[str]) -> str:
    domain = urlparse(source_url).netloc.lower() if source_url else ""
    cleaned = title.strip()
    query = f"\"{cleaned}\""
    if domain:
        query = f"{query} -site:{domain}"
    return _truncate_query(query)


def _search_brave_references(query: str, max_results: int) -> tuple[List[SourceCandidate], Optional[int]]:
    payload = _search_brave_payload(query, max_results)
    total = payload.get("web", {}).get("total")
    results: List[SourceCandidate] = []
    for item in payload.get("web", {}).get("results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
            )
        )
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None
    return results, total_int


def _search_serpapi_references(query: str, max_results: int) -> tuple[List[SourceCandidate], Optional[int]]:
    payload = _search_serpapi_payload(query, max_results)
    total = payload.get("search_information", {}).get("total_results")
    results: List[SourceCandidate] = []
    for item in payload.get("organic_results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("link", ""),
                url=item.get("link", ""),
            )
        )
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None
    return results, total_int


def _reference_count_for_query(query: str, max_results: int) -> tuple[int, int]:
    if not query:
        return 0, 0
    results, total = _search_serpapi_references(query, max_results)
    count = total if total is not None else len(results)
    return max(count, 0), max(len(results), 0)


@tool
def count_references(
    source_url: Optional[str] = None,
    title: Optional[str] = None,
    published_at: Optional[str] = None,
    max_results: int = REFERENCE_SEARCH_MAX_RESULTS,
) -> dict:
    """Count web references to a source URL or title."""
    if not source_url and not title:
        return {
            "coverage_count": 0,
            "url_count": 0,
            "title_count": 0,
            "result_count": 0,
            "url_query": "",
            "title_query": "",
        }
    _ = _parse_iso_datetime(published_at)
    url_query = _build_url_reference_query(source_url) if source_url else ""
    title_query = _build_title_reference_query(title, source_url) if title else ""
    url_count, url_results = _reference_count_for_query(url_query, max_results)
    title_count, title_results = _reference_count_for_query(title_query, max_results)
    coverage_count = max(url_count, title_count)
    result_count = max(url_results, title_results)
    return {
        "coverage_count": coverage_count,
        "url_count": url_count,
        "title_count": title_count,
        "result_count": result_count,
        "url_query": url_query,
        "title_query": title_query,
    }


@tool
def fetch_feed(feed_url: str, lookback_days: int = 2, source_name: Optional[str] = None) -> List[dict]:
    """Fetch announcements from an RSS/Atom/JSON feed."""
    response = requests.get(feed_url, timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()

    items: List[SourceItem] = []
    if "json" in content_type or feed_url.endswith(".json"):
        payload = response.json()
        for entry in payload.get("items", []):
            published_raw = entry.get("date_published") or entry.get("date_modified")
            published = _parse_iso_datetime(published_raw) if published_raw else None
            items.append(
                SourceItem(
                    title=entry.get("title") or "Untitled",
                    url=entry.get("url") or entry.get("external_url") or feed_url,
                    published_at=published,
                    source=source_name or payload.get("title") or feed_url,
                    summary=entry.get("summary") or entry.get("content_text"),
                )
            )
        return [item.model_dump() for item in _filter_recent(items, lookback_days)]

    feed = feedparser.parse(response.text)
    feed_title = feed.feed.get("title") if hasattr(feed, "feed") else None
    for entry in feed.entries:
        published = _parse_datetime(entry.get("published_parsed") or entry.get("updated_parsed"))
        items.append(
            SourceItem(
                title=entry.get("title", "Untitled"),
                url=entry.get("link", feed_url),
                published_at=published,
                source=source_name or feed_title or feed_url,
                summary=entry.get("summary"),
            )
        )
    return [item.model_dump() for item in _filter_recent(items, lookback_days)]
