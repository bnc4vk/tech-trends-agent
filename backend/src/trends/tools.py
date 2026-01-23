from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from .schemas import FeedCandidate, SourceCandidate, SourceItem

DEFAULT_TIMEOUT = 20
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS"))
SEARCH_MAX_QUERY_CHARS = int(os.getenv("SEARCH_MAX_QUERY_CHARS"))
REFERENCE_SEARCH_MAX_RESULTS = int(os.getenv("REFERENCE_SEARCH_MAX_RESULTS", "8"))
REFERENCE_SEARCH_DEPTH = os.getenv("REFERENCE_SEARCH_DEPTH", "basic")

FEED_HINTS = [
    "feed",
    "rss",
    "atom.xml",
    "feed.xml",
    "rss.xml",
    "index.xml",
    "blog/rss.xml",
    "blog/feed.xml",
]


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime(*value[:6])
    except Exception:
        return None


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _filter_recent(items: List[SourceItem], lookback_days: int) -> List[SourceItem]:
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    return [item for item in items if not item.published_at or item.published_at >= cutoff]


def _truncate_query(query: str) -> str:
    """Truncate query to fit within character limit, preserving word boundaries."""
    if len(query) <= SEARCH_MAX_QUERY_CHARS:
        return query
    # Truncate and find last space to avoid cutting words
    truncated = query[:SEARCH_MAX_QUERY_CHARS]
    last_space = truncated.rfind(" ")
    if last_space > SEARCH_MAX_QUERY_CHARS * 0.8:  # Only use if we keep most of the query
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
                snippet=item.get("content"),
                score=item.get("score"),
            )
        )
    return results


def _search_brave(query: str, max_results: int) -> List[SourceCandidate]:
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
    payload = response.json()
    results: List[SourceCandidate] = []
    for item in payload.get("web", {}).get("results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=item.get("description"),
                score=None,
            )
        )
    return results


def _search_serpapi(query: str, max_results: int) -> List[SourceCandidate]:
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
    payload = response.json()
    results: List[SourceCandidate] = []
    for item in payload.get("organic_results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("link", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet"),
                score=None,
            )
        )
    return results


def _dedupe_sources(candidates: List[SourceCandidate]) -> List[SourceCandidate]:
    seen = set()
    deduped: List[SourceCandidate] = []
    for candidate in candidates:
        domain = urlparse(candidate.url).netloc.lower()
        if not domain or domain in seen:
            continue
        seen.add(domain)
        deduped.append(candidate)
    return deduped


def _build_reference_query(source_url: str, published_at: Optional[datetime]) -> str:
    domain = urlparse(source_url).netloc.lower()
    query = f"\"{source_url}\""
    if domain:
        query = f"{query} -site:{domain}"
    if published_at:
        query = f"{query} since {published_at.date().isoformat()}"
    return _truncate_query(query)


def _search_brave_references(query: str, max_results: int) -> tuple[List[SourceCandidate], Optional[int]]:
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
    payload = response.json()
    total = payload.get("web", {}).get("total")
    results: List[SourceCandidate] = []
    for item in payload.get("web", {}).get("results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=item.get("description"),
                score=None,
            )
        )
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None
    return results, total_int


def _search_serpapi_references(query: str, max_results: int) -> tuple[List[SourceCandidate], Optional[int]]:
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
    payload = response.json()
    total = payload.get("search_information", {}).get("total_results")
    results: List[SourceCandidate] = []
    for item in payload.get("organic_results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("link", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet"),
                score=None,
            )
        )
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None
    return results, total_int


@tool
def search_sources(domain_description: str, max_results: int = SEARCH_MAX_RESULTS) -> List[dict]:
    """Search for high-signal publications and sources for a domain description."""
    # Condense query to focus on finding publications with feeds
    # Extract key terms from domain_description to keep it concise
    query = f"{domain_description} publications blogs newsletters RSS feeds"
    # Truncate before constructing full query to ensure we stay under limit
    query = _truncate_query(query)
    if SEARCH_PROVIDER == "brave":
        results = _search_brave(query, max_results)
    elif SEARCH_PROVIDER == "serpapi":
        results = _search_serpapi(query, max_results)
    else:
        results = _search_tavily(query, max_results)
    return [candidate.model_dump() for candidate in _dedupe_sources(results)]


@tool
def count_references(
    source_url: str, published_at: Optional[str] = None, max_results: int = REFERENCE_SEARCH_MAX_RESULTS
) -> dict:
    """Count web references to a source URL."""
    if not source_url:
        return {"reference_count": 0, "result_count": 0, "query": ""}
    published = _parse_iso_datetime(published_at)
    query = _build_reference_query(source_url, published)
    if SEARCH_PROVIDER == "brave":
        results, total = _search_brave_references(query, max_results)
        count = total if total is not None else len(results)
    elif SEARCH_PROVIDER == "serpapi":
        results, total = _search_serpapi_references(query, max_results)
        count = total if total is not None else len(results)
    else:
        results = _search_tavily(query, max_results, search_depth=REFERENCE_SEARCH_DEPTH)
        count = len(results)
    return {"reference_count": count, "result_count": len(results), "query": query}


def _guess_feed_urls(source_url: str) -> List[str]:
    parsed = urlparse(source_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return [urljoin(base + "/", hint) for hint in FEED_HINTS]


def _extract_feed_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel", [])).lower()
        link_type = (link.get("type") or "").lower()
        href = link.get("href")
        if not href:
            continue
        if "alternate" in rel and any(token in link_type for token in ["rss", "atom", "json"]):
            candidates.append(urljoin(base_url, href))
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        text = (anchor.get_text() or "").lower()
        if any(token in text for token in ["rss", "atom", "feed"]) or any(
            token in href.lower() for token in ["rss", "atom", "feed"]
        ):
            candidates.append(urljoin(base_url, href))
    return candidates


def _validate_feed_url(feed_url: str) -> Optional[FeedCandidate]:
    try:
        response = requests.get(feed_url, timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS, allow_redirects=True)
        if not response.ok:
            return None
    except requests.RequestException:
        return None

    content_type = response.headers.get("content-type", "").lower()
    text = response.text

    if "json" in content_type or feed_url.endswith(".json"):
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "items" in payload:
            return FeedCandidate(feed_url=feed_url, feed_type="json", title=payload.get("title"))
        return None

    feed = feedparser.parse(text)
    if feed.entries:
        feed_title = feed.feed.get("title") if hasattr(feed, "feed") else None
        return FeedCandidate(feed_url=feed_url, feed_type="rss", title=feed_title)
    return None


def _discover_feeds_impl(source_url: str, max_feeds: int) -> List[FeedCandidate]:
    try:
        response = requests.get(source_url, timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS, allow_redirects=True)
        response.raise_for_status()
    except requests.HTTPError as exc:
        # For 403/404, try to guess feed URLs from domain without fetching the page
        if exc.response.status_code in (403, 404):
            candidates = _guess_feed_urls(source_url)
            validated: List[FeedCandidate] = []
            seen = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                feed = _validate_feed_url(candidate)
                if feed:
                    validated.append(feed)
                if len(validated) >= max_feeds:
                    break
            return validated
        raise
    
    content_type = response.headers.get("content-type", "").lower()
    looks_like_xml = response.text.lstrip().startswith("<?xml")
    if any(token in content_type for token in ["xml", "rss", "atom"]) or looks_like_xml:
        candidates = [response.url, source_url]
    else:
        candidates = _extract_feed_links(response.text, source_url)
    candidates.extend(_guess_feed_urls(source_url))

    validated: List[FeedCandidate] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        feed = _validate_feed_url(candidate)
        if feed:
            validated.append(feed)
        if len(validated) >= max_feeds:
            break
    return validated


@tool
def discover_feeds(source_url: str, max_feeds: int = 3) -> List[dict]:
    """Discover RSS/Atom/JSON feeds for a publication or source."""
    return [feed.model_dump() for feed in _discover_feeds_impl(source_url, max_feeds)]


@tool
def standardize_source(source_url: str, max_feeds: int = 3) -> List[dict]:
    """Convert a source into standardized feed endpoints (RSS/Atom/JSON)."""
    return [feed.model_dump() for feed in _discover_feeds_impl(source_url, max_feeds)]


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
            published = None
            if published_raw:
                try:
                    published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                except ValueError:
                    published = None
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
