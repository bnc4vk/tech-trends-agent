from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urldefrag, urlparse

import feedparser
import requests
from langchain_core.tools import tool

from .config import (
    BRAVE_SEARCH_API_KEY,
    DYNAMIC_DISCOVERY_ENABLED,
    DYNAMIC_DISCOVERY_MAX_QUERIES,
    DYNAMIC_DISCOVERY_MAX_RESULTS_PER_QUERY,
    DYNAMIC_DISCOVERY_MAX_TOTAL_RESULTS,
    DYNAMIC_DISCOVERY_PER_CATEGORY,
    DYNAMIC_DISCOVERY_SEARCH_DEPTH,
    REFERENCE_SEARCH_MAX_RESULTS,
    SEARCH_MAX_QUERY_CHARS,
    SEARCH_PROVIDER,
    SERPAPI_API_KEY,
    TAVILY_API_KEY,
)
from .schemas import (
    Category,
    DynamicDiscoveryMetadata,
    DynamicDiscoveryResult,
    SourceCandidate,
    SourceItem,
)

DEFAULT_TIMEOUT = 20
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

DYNAMIC_DISCOVERY_EXCLUDED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "reddit.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
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
                summary=item.get("content"),
                published_at=_parse_iso_datetime(item.get("published_date")),
                source=urlparse(item.get("url", "")).netloc,
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
                summary=item.get("description"),
                source=item.get("profile", {}).get("name") or urlparse(item.get("url", "")).netloc,
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
                summary=item.get("snippet"),
                published_at=_parse_iso_datetime(item.get("date")),
                source=item.get("source") or item.get("displayed_link") or urlparse(item.get("link", "")).netloc,
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


def _dynamic_query(category: Category, lookback_days: int) -> str:
    after = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()
    category_terms = {
        "product": "technology product launch AI developer tools major update",
        "research": "AI research breakthrough paper benchmark machine learning",
        "infra": "cloud infrastructure chips developer platform framework release",
    }
    return _truncate_query(f"{category_terms[category]} after:{after}")


def _normalize_provider(provider: Optional[str] = None) -> str:
    normalized = (provider or SEARCH_PROVIDER or "tavily").strip().lower()
    aliases = {
        "brave_search": "brave",
        "brave-search": "brave",
        "serp": "serpapi",
        "serp_api": "serpapi",
    }
    return aliases.get(normalized, normalized)


def _search_provider_skip_reason(provider: str) -> Optional[str]:
    if provider == "tavily":
        return None if TAVILY_API_KEY else "TAVILY_API_KEY is not configured."
    if provider == "brave":
        return None if BRAVE_SEARCH_API_KEY else "BRAVE_SEARCH_API_KEY is not configured."
    if provider == "serpapi":
        return None if SERPAPI_API_KEY else "SERPAPI_API_KEY is not configured."
    return f"Unsupported SEARCH_PROVIDER '{provider}'."


def _normalize_url_key(url: str) -> str:
    if not url:
        return ""
    trimmed = url.strip().lower()
    trimmed, _ = urldefrag(trimmed)
    if trimmed.endswith("/"):
        trimmed = trimmed[:-1]
    return trimmed


def _is_excluded_dynamic_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return any(domain == blocked or domain.endswith(f".{blocked}") for blocked in DYNAMIC_DISCOVERY_EXCLUDED_DOMAINS)


def _candidate_source_name(candidate: SourceCandidate, provider: str) -> str:
    if candidate.source:
        return candidate.source
    domain = urlparse(candidate.url).netloc
    return domain or f"Dynamic Search ({provider})"


def _search_candidates(query: str, max_results: int, provider: Optional[str] = None) -> List[SourceCandidate]:
    provider_name = _normalize_provider(provider)
    if provider_name == "brave":
        results, _total = _search_brave_references(query, max_results)
        return results
    if provider_name == "serpapi":
        results, _total = _search_serpapi_references(query, max_results)
        return results
    if provider_name == "tavily":
        return _search_tavily(query, max_results, search_depth=DYNAMIC_DISCOVERY_SEARCH_DEPTH)
    raise RuntimeError(f"Unsupported SEARCH_PROVIDER '{provider_name}'.")


def _bounded_discovery_metadata(
    enabled: bool,
    provider: str,
    requested_queries: int,
    max_queries: int,
    max_results_per_query: int,
    max_total_results: int,
    skipped_reason: Optional[str] = None,
) -> DynamicDiscoveryMetadata:
    return DynamicDiscoveryMetadata(
        enabled=enabled,
        provider=provider,
        requested_queries=requested_queries,
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
        max_total_results=max_total_results,
        skipped_reason=skipped_reason,
    )


def _discover_trending_candidates(
    category: Category,
    lookback_days: int,
    max_results: int,
    queries: Optional[List[str]] = None,
    provider: Optional[str] = None,
) -> DynamicDiscoveryResult:
    provider_name = _normalize_provider(provider)
    raw_queries = queries or [_dynamic_query(category, lookback_days)]
    max_queries = max(1, DYNAMIC_DISCOVERY_MAX_QUERIES)
    max_results_per_query = max(1, min(max_results, DYNAMIC_DISCOVERY_MAX_RESULTS_PER_QUERY))
    max_total_results = max(1, min(max_results, DYNAMIC_DISCOVERY_MAX_TOTAL_RESULTS))

    metadata = _bounded_discovery_metadata(
        enabled=DYNAMIC_DISCOVERY_ENABLED,
        provider=provider_name,
        requested_queries=len(raw_queries),
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
        max_total_results=max_total_results,
    )

    if not DYNAMIC_DISCOVERY_ENABLED:
        metadata.skipped_reason = "DYNAMIC_DISCOVERY_ENABLED is false."
        return DynamicDiscoveryResult(metadata=metadata)

    skip_reason = _search_provider_skip_reason(provider_name)
    if skip_reason:
        metadata.skipped_reason = skip_reason
        return DynamicDiscoveryResult(metadata=metadata)

    bounded_queries = []
    seen_queries: set[str] = set()
    for raw_query in raw_queries:
        query = _truncate_query(raw_query.strip())
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        bounded_queries.append(query)
        if len(bounded_queries) >= max_queries:
            break

    metadata.requested_queries = len(raw_queries)
    discovered: List[SourceItem] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for query in bounded_queries:
        if len(discovered) >= max_total_results:
            break
        remaining = max_total_results - len(discovered)
        query_limit = min(max_results_per_query, remaining)
        try:
            candidates = _search_candidates(query, query_limit, provider=provider_name)
            metadata.executed_queries += 1
        except Exception as exc:
            metadata.errors.append(f"{query}: {exc}")
            continue

        for rank, candidate in enumerate(candidates, start=1):
            if len(discovered) >= max_total_results:
                break
            if not candidate.url:
                continue
            if _is_excluded_dynamic_domain(candidate.url):
                continue
            url_key = _normalize_url_key(candidate.url)
            title_key = candidate.title.strip().lower()
            if (url_key and url_key in seen_urls) or (title_key and title_key in seen_titles):
                continue
            if url_key:
                seen_urls.add(url_key)
            if title_key:
                seen_titles.add(title_key)
            discovered.append(
                SourceItem(
                    title=candidate.title or candidate.url,
                    url=candidate.url,
                    published_at=candidate.published_at,
                    source=_candidate_source_name(candidate, provider_name),
                    summary=candidate.summary,
                    category=category,
                    discovery_method="search",
                    discovery_query=query,
                    discovery_provider=provider_name,
                    discovery_rank=rank,
                )
            )

    metadata.result_count = len(discovered)
    metadata.dynamic_discovery_count = len(discovered)
    return DynamicDiscoveryResult(items=discovered, metadata=metadata)


@tool
def discover_trending_candidates(
    category: Category,
    lookback_days: int = 3,
    max_results: int = DYNAMIC_DISCOVERY_PER_CATEGORY,
    queries: Optional[List[str]] = None,
) -> List[dict]:
    """Discover recent technology trend candidates via the configured search provider."""
    result = _discover_trending_candidates(
        category=category,
        lookback_days=lookback_days,
        max_results=max_results,
        queries=queries,
    )
    return [item.model_dump() for item in result.items]


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
