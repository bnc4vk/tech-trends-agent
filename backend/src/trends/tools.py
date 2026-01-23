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


def _search_tavily(query: str, max_results: int) -> List[SourceCandidate]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not configured.")
    query = _truncate_query(query)
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
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


def _fetch_rss(feed_url: str, source_name: str, lookback_days: int) -> List[SourceItem]:
    response = requests.get(feed_url, timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    feed = feedparser.parse(response.text)
    items: List[SourceItem] = []
    for entry in feed.entries:
        published = _parse_datetime(entry.get("published_parsed"))
        items.append(
            SourceItem(
                title=entry.get("title", "Untitled"),
                url=entry.get("link", feed_url),
                published_at=published,
                source=source_name,
                summary=entry.get("summary"),
            )
        )
    return _filter_recent(items, lookback_days)


def _fetch_github_trending(lookback_days: int) -> List[SourceItem]:
    url = "https://github.com/trending"
    response = requests.get(url, timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    items: List[SourceItem] = []

    for repo in soup.select("article.Box-row"):
        name = repo.select_one("h2 a")
        if not name:
            continue
        title = " ".join(name.text.split())
        href = name.get("href", "")
        summary = repo.select_one("p")
        items.append(
            SourceItem(
                title=title,
                url=f"https://github.com{href}",
                published_at=None,
                source="GitHub Trending",
                summary=summary.text.strip() if summary else None,
            )
        )
    return _filter_recent(items, lookback_days)


@tool
def bens_bites(lookback_days: int = 2) -> List[dict]:
    """Fetch Ben's Bites (daily AI roundup)."""
    items = _fetch_rss("https://www.bensbites.co/feed", "Ben's Bites", lookback_days)
    return [item.model_dump() for item in items]


@tool
def the_batch(lookback_days: int = 7) -> List[dict]:
    """Fetch The Batch (DeepLearning.AI)."""
    items = _fetch_rss("https://www.deeplearning.ai/the-batch/feed/", "The Batch", lookback_days)
    return [item.model_dump() for item in items]


@tool
def stratechery(lookback_days: int = 7) -> List[dict]:
    """Fetch Stratechery articles."""
    items = _fetch_rss("https://stratechery.com/feed/", "Stratechery", lookback_days)
    return [item.model_dump() for item in items]


@tool
def latent_space(lookback_days: int = 7) -> List[dict]:
    """Fetch Latent Space posts."""
    items = _fetch_rss("https://www.latent.space/feed", "Latent Space", lookback_days)
    return [item.model_dump() for item in items]


@tool
def arxiv_sanity(lookback_days: int = 2) -> List[dict]:
    """Fetch arXiv Sanity feed."""
    items = _fetch_rss("http://www.arxiv-sanity.com/rss", "arXiv Sanity", lookback_days)
    return [item.model_dump() for item in items]


@tool
def papers_with_code(lookback_days: int = 2) -> List[dict]:
    """Fetch Papers with Code trending."""
    items = _fetch_rss("https://paperswithcode.com/rss", "Papers with Code", lookback_days)
    return [item.model_dump() for item in items]


@tool
def huggingface_trending(lookback_days: int = 2) -> List[dict]:
    """Fetch Hugging Face trending models/datasets."""
    items = _fetch_rss("https://huggingface.co/blog/feed.xml", "Hugging Face Trending", lookback_days)
    return [item.model_dump() for item in items]


@tool
def huggingface_daily_papers(lookback_days: int = 2) -> List[dict]:
    """Fetch Hugging Face daily papers."""
    items = _fetch_rss("https://huggingface.co/papers?format=rss", "Hugging Face Daily Papers", lookback_days)
    return [item.model_dump() for item in items]


@tool
def openai_blog(lookback_days: int = 7) -> List[dict]:
    """Fetch OpenAI blog."""
    items = _fetch_rss("https://openai.com/blog/rss", "OpenAI Blog", lookback_days)
    return [item.model_dump() for item in items]


@tool
def anthropic_blog(lookback_days: int = 7) -> List[dict]:
    """Fetch Anthropic blog."""
    items = _fetch_rss("https://www.anthropic.com/news/rss.xml", "Anthropic Blog", lookback_days)
    return [item.model_dump() for item in items]


@tool
def deepmind_blog(lookback_days: int = 7) -> List[dict]:
    """Fetch Google DeepMind blog."""
    items = _fetch_rss("https://deepmind.google/blog/rss.xml", "Google DeepMind Blog", lookback_days)
    return [item.model_dump() for item in items]


@tool
def semianalysis(lookback_days: int = 7) -> List[dict]:
    """Fetch SemiAnalysis posts."""
    items = _fetch_rss("https://semianalysis.com/feed", "SemiAnalysis", lookback_days)
    return [item.model_dump() for item in items]


@tool
def github_trending(lookback_days: int = 2) -> List[dict]:
    """Fetch GitHub trending repositories."""
    items = _fetch_github_trending(lookback_days)
    return [item.model_dump() for item in items]


@tool
def github_release_radar(lookback_days: int = 7) -> List[dict]:
    """Fetch GitHub Release Radar posts."""
    items = _fetch_rss("https://github.blog/feed/", "GitHub Release Radar", lookback_days)
    return [item.model_dump() for item in items]


@tool
def simon_willison_blog(lookback_days: int = 7) -> List[dict]:
    """Fetch Simon Willison's blog."""
    items = _fetch_rss("https://simonwillison.net/atom/everything/", "Simon Willison", lookback_days)
    return [item.model_dump() for item in items]


@tool
def deno_changelog(lookback_days: int = 7) -> List[dict]:
    """Fetch Deno changelog."""
    items = _fetch_rss("https://deno.com/feed", "Deno Changelog", lookback_days)
    return [item.model_dump() for item in items]


@tool
def vercel_changelog(lookback_days: int = 7) -> List[dict]:
    """Fetch Vercel changelog."""
    items = _fetch_rss("https://vercel.com/changelog/rss.xml", "Vercel Changelog", lookback_days)
    return [item.model_dump() for item in items]


@tool
def supabase_changelog(lookback_days: int = 7) -> List[dict]:
    """Fetch Supabase changelog."""
    items = _fetch_rss("https://supabase.com/blog/rss.xml", "Supabase Changelog", lookback_days)
    return [item.model_dump() for item in items]


@tool
def ai_engineer_summit(lookback_days: int = 30) -> List[dict]:
    """Fetch AI Engineer Summit talks + repos."""
    items = _fetch_rss("https://www.ai.engineer/feed.xml", "AI Engineer Summit", lookback_days)
    return [item.model_dump() for item in items]


STATIC_FEED_TOOLS = [
    bens_bites,
    the_batch,
    stratechery,
    latent_space,
    arxiv_sanity,
    papers_with_code,
    huggingface_trending,
    huggingface_daily_papers,
    openai_blog,
    anthropic_blog,
    deepmind_blog,
    semianalysis,
    github_trending,
    github_release_radar,
    simon_willison_blog,
    deno_changelog,
    vercel_changelog,
    supabase_changelog,
    ai_engineer_summit,
]

# Backwards-compat alias for legacy collection flows.
TREND_TOOLS = STATIC_FEED_TOOLS
