from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from .config import (
    BRAVE_SEARCH_API_KEY,
    REFERENCE_SEARCH_DEPTH,
    REFERENCE_SEARCH_MAX_RESULTS,
    SEARCH_MAX_QUERY_CHARS,
    SEARCH_MAX_RESULTS,
    SEARCH_PROVIDER,
    SERPAPI_API_KEY,
    TAVILY_API_KEY,
)
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

NON_RSS_CLASS_HINTS = [
    "post",
    "entry",
    "article",
    "story",
    "card",
    "news",
    "item",
    "blog",
]

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
]

URL_DATE_RE = re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:/|$)")


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


def _parse_date_value(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    parsed = _parse_iso_datetime(cleaned)
    if parsed:
        return parsed
    for fmt in DATE_FORMATS:
        try:
            return _to_naive_utc(datetime.strptime(cleaned, fmt))
        except ValueError:
            continue
    return None


def _parse_date_from_url(url: str) -> Optional[datetime]:
    if not url:
        return None
    match = URL_DATE_RE.search(url)
    if not match:
        return None
    try:
        year, month, day = (int(part) for part in match.groups())
        return datetime(year, month, day)
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


def _filter_recent_strict(items: List[SourceItem], lookback_days: int) -> List[SourceItem]:
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    filtered: List[SourceItem] = []
    for item in items:
        if item.published_at:
            item.published_at = _to_naive_utc(item.published_at)
        if item.published_at and item.published_at >= cutoff:
            filtered.append(item)
    return filtered


def _has_class_hint(value: Optional[str | list]) -> bool:
    if not value:
        return False
    if isinstance(value, (list, tuple)):
        text = " ".join(value)
    else:
        text = str(value)
    text = text.lower()
    return any(token in text for token in NON_RSS_CLASS_HINTS)


def _extract_date_from_node(node: BeautifulSoup) -> Optional[datetime]:
    time_tag = node.find("time")
    if time_tag:
        for attr in ("datetime", "content"):
            parsed = _parse_date_value(time_tag.get(attr))
            if parsed:
                return parsed
        parsed = _parse_date_value(time_tag.get_text(" ", strip=True))
        if parsed:
            return parsed
    return None


def _extract_summary(node: BeautifulSoup) -> Optional[str]:
    summary_tag = node.find("p")
    if not summary_tag:
        return None
    text = summary_tag.get_text(" ", strip=True)
    if not text:
        return None
    return text[:280]


def _extract_non_rss_items(
    html: str, base_url: str, source_name: str, max_items: int
) -> List[SourceItem]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[SourceItem] = []
    seen_urls: set[str] = set()

    def add_candidate(title: str, url: str, published_at: Optional[datetime], summary: Optional[str]) -> None:
        if not title or not url or url in seen_urls or not published_at:
            return
        seen_urls.add(url)
        candidates.append(
            SourceItem(
                title=title,
                url=url,
                published_at=published_at,
                source=source_name,
                summary=summary,
            )
        )

    nodes = soup.find_all("article")
    if not nodes:
        nodes = soup.find_all(["div", "li", "section"], class_=_has_class_hint)

    for node in nodes:
        if len(candidates) >= max_items:
            break
        link = node.find("a", href=True)
        if not link:
            continue
        href = link.get("href")
        if not href or href.startswith("#") or href.startswith("mailto:") or "javascript:" in href:
            continue
        url = urljoin(base_url, href)
        title_tag = node.find(["h1", "h2", "h3"])
        title = (
            title_tag.get_text(" ", strip=True)
            if title_tag and title_tag.get_text(strip=True)
            else link.get_text(" ", strip=True)
        )
        if not title:
            continue
        published_at = _extract_date_from_node(node) or _parse_date_from_url(url)
        summary = _extract_summary(node)
        add_candidate(title, url, published_at, summary)

    if not candidates:
        links = soup.select("main h2 a, main h3 a, article h2 a, article h3 a")
        for link in links:
            if len(candidates) >= max_items:
                break
            href = link.get("href")
            if not href or href.startswith("#") or href.startswith("mailto:") or "javascript:" in href:
                continue
            url = urljoin(base_url, href)
            title = link.get_text(" ", strip=True)
            if not title:
                continue
            published_at = _parse_date_from_url(url)
            add_candidate(title, url, published_at, None)

    return candidates


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


def _search_brave(query: str, max_results: int) -> List[SourceCandidate]:
    payload = _search_brave_payload(query, max_results)
    results: List[SourceCandidate] = []
    for item in payload.get("web", {}).get("results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
            )
        )
    return results


def _search_serpapi(query: str, max_results: int) -> List[SourceCandidate]:
    payload = _search_serpapi_payload(query, max_results)
    results: List[SourceCandidate] = []
    for item in payload.get("organic_results", []):
        results.append(
            SourceCandidate(
                title=item.get("title") or item.get("link", ""),
                url=item.get("link", ""),
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
def fetch_non_rss(
    source_url: str,
    lookback_days: int = 2,
    source_name: Optional[str] = None,
    max_items: int = 20,
) -> List[dict]:
    """Fetch recent posts from a non-RSS source page using HTML heuristics."""
    if not source_url:
        return []
    response = requests.get(source_url, timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    items = _extract_non_rss_items(response.text, source_url, source_name or source_url, max_items)
    return [item.model_dump() for item in _filter_recent_strict(items, lookback_days)]


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
