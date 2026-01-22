from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from .schemas import SourceItem

DEFAULT_TIMEOUT = 20


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


def _fetch_rss(feed_url: str, source_name: str, lookback_days: int) -> List[SourceItem]:
    feed = feedparser.parse(feed_url)
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
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
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


TREND_TOOLS = [
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
