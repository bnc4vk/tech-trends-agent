from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, date
from typing import List
from urllib.parse import urldefrag

from langgraph.graph import StateGraph

from .agents import build_experts, evaluate_items, screen_items
from .config import DEFAULT_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS, OVERWRITE_EXECUTION, TARGET_TRENDS
from .schemas import GraphState, SourceItem, TrendItem
from .scoring import compute_trending_score
from .supabase_store import daily_record_exists, upsert_daily_record
from .tools import count_references, discover_feeds, fetch_feed, search_sources


VERBOSE = os.getenv("TRENDS_VERBOSE")
LOOKBACK_STEP_DAYS = int(os.getenv("LOOKBACK_STEP_DAYS"))
MAX_SOURCES_PER_EXPERT = int(os.getenv("MAX_SOURCES_PER_EXPERT"))
MAX_FEEDS_PER_SOURCE = int(os.getenv("MAX_FEEDS_PER_SOURCE"))
MAX_ITEMS_PER_EXPERT = int(os.getenv("MAX_ITEMS_PER_EXPERT", "30"))
MAX_TRENDS_PER_CATEGORY = int(os.getenv("MAX_TRENDS_PER_CATEGORY", "12"))
MAX_REFERENCE_LOOKUPS = int(os.getenv("MAX_REFERENCE_LOOKUPS", "30"))

def _log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


def _normalize_title(title: str) -> str:
    return "".join(ch for ch in title.lower() if ch.isalnum() or ch.isspace()).strip()


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    trimmed = url.strip().lower()
    trimmed, _ = urldefrag(trimmed)
    if trimmed.endswith("/"):
        trimmed = trimmed[:-1]
    return trimmed


def _hash_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def collect_sources(state: GraphState) -> GraphState:
    items: List[SourceItem] = []
    errors = list(state.errors)
    experts = build_experts()
    feed_urls: set[str] = set()
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    title_references: dict[str, set[str]] = {}

    _log(f"[collect] Starting discovery with {len(experts)} experts...")
    for expert in experts:
        expert_items = 0
        try:
            start = time.perf_counter()
            _log(f"[collect] -> search_sources ({expert.name})")
            sources = search_sources.invoke(
                {"domain_description": expert.domain_description, "max_results": MAX_SOURCES_PER_EXPERT}
            )
            elapsed = time.perf_counter() - start
            _log(f"[collect] <- search_sources ({expert.name}) {len(sources)} sources, {elapsed:.1f}s")
        except Exception as exc:
            errors.append(f"search_sources/{expert.name}: {exc}")
            _log(f"[collect] !! search_sources ({expert.name}) failed: {exc}")
            sources = []

        for candidate in sources:
            if expert_items >= MAX_ITEMS_PER_EXPERT:
                break
            source_url = candidate.get("url")
            if not source_url:
                continue
            try:
                feeds = discover_feeds.invoke({"source_url": source_url, "max_feeds": MAX_FEEDS_PER_SOURCE})
                if not feeds:
                    _log(f"[collect] No feeds found for {source_url}")
            except Exception as exc:
                errors.append(f"discover_feeds/{source_url}: {exc}")
                _log(f"[collect] !! discover_feeds failed for {source_url}: {exc}")
                continue

            feed_count = 0
            item_count_before = len(items)
            for feed in feeds:
                if expert_items >= MAX_ITEMS_PER_EXPERT:
                    break
                feed_url = feed.get("feed_url")
                if not feed_url or feed_url in feed_urls:
                    continue
                feed_urls.add(feed_url)
                try:
                    results = fetch_feed.invoke(
                        {
                            "feed_url": feed_url,
                            "lookback_days": state.lookback_days,
                            "source_name": candidate.get("title") or feed.get("title"),
                        }
                    )
                    feed_count += 1
                    for raw in results:
                        item = SourceItem(**raw)
                        title_key = _normalize_title(item.title)
                        if title_key:
                            title_references.setdefault(title_key, set()).add(item.source)
                        url_key = _normalize_url(item.url)
                        if (title_key and title_key in seen_titles) or (url_key and url_key in seen_urls):
                            continue
                        if title_key:
                            seen_titles.add(title_key)
                        if url_key:
                            seen_urls.add(url_key)
                        items.append(item)
                        expert_items += 1
                        if expert_items >= MAX_ITEMS_PER_EXPERT:
                            break
                except Exception as exc:
                    errors.append(f"fetch_feed/{feed_url}: {exc}")
                    _log(f"[collect] !! fetch_feed failed for {feed_url}: {exc}")
            
            items_added = len(items) - item_count_before
            if feed_count > 0 and items_added == 0:
                _log(f"[collect] Found {feed_count} feed(s) for {source_url} but 0 items within {state.lookback_days} day lookback")
        if expert_items >= MAX_ITEMS_PER_EXPERT:
            _log(f"[collect] {expert.name} reached {MAX_ITEMS_PER_EXPERT} items; stopping early.")

    _log(f"[collect] Done. Total items: {len(items)}")
    serialized_refs = {key: sorted(list(values)) for key, values in title_references.items()}
    return state.model_copy(update={"raw_items": items, "errors": errors, "title_references": serialized_refs})


def screen_sources(state: GraphState) -> GraphState:
    if not state.raw_items:
        return state
    _log(f"[screen] Reviewing {len(state.raw_items)} items for relevance...")
    screened = screen_items(state.raw_items)
    if not screened:
        _log("[screen] All items discarded.")
        return state.model_copy(update={"raw_items": []})
    keep_titles = {_normalize_title(item.title) for item in screened}
    filtered_refs = {
        key: sources for key, sources in state.title_references.items() if key in keep_titles
    }
    _log(f"[screen] Kept {len(screened)} items after screening.")
    return state.model_copy(update={"raw_items": screened, "title_references": filtered_refs})


def review_lookback(state: GraphState) -> GraphState:
    target = state.target_trend_count
    if len(state.raw_items) >= target:
        _log(f"[review] Target met ({len(state.raw_items)}/{target}).")
        return state
    if state.lookback_days >= state.max_lookback_days:
        _log(f"[review] Max lookback reached ({state.lookback_days} days).")
        return state

    new_lookback = min(state.lookback_days + LOOKBACK_STEP_DAYS, state.max_lookback_days)
    if new_lookback == state.lookback_days:
        return state
    history = list(state.lookback_history)
    history.append(new_lookback)
    _log(f"[review] Expanding lookback to {new_lookback} days to reach ~{target} items.")
    return state.model_copy(update={"lookback_days": new_lookback, "lookback_history": history})


def should_expand(state: GraphState) -> str:
    if len(state.raw_items) < state.target_trend_count and state.lookback_days < state.max_lookback_days:
        return "collect"
    return "evaluate"


def evaluate_sources(state: GraphState) -> GraphState:
    _log(f"[evaluate] Scoring {len(state.raw_items)} items...")
    assessed = evaluate_items(state.raw_items)
    trends: List[TrendItem] = []

    title_groups: dict[str, List[str]] = {}
    if state.title_references:
        title_groups = {key: list(sources) for key, sources in state.title_references.items()}
    else:
        for item in state.raw_items:
            key = _normalize_title(item.title)
            title_groups.setdefault(key, []).append(item.source)

    reference_cache: dict[str, int] = {}
    remaining_budget = MAX_REFERENCE_LOOKUPS
    prioritized = sorted(
        assessed,
        key=lambda x: (
            x[1].impact_score,
            -((datetime.utcnow() - x[0].published_at).days if x[0].published_at else 9999),
        ),
        reverse=True,
    )

    for item, assessment, _category in prioritized:
        url_key = _normalize_url(item.url)
        if not url_key or url_key in reference_cache:
            continue
        if remaining_budget <= 0:
            break
        try:
            payload = count_references.invoke(
                {
                    "source_url": item.url,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                }
            )
            reference_cache[url_key] = int(payload.get("reference_count", 0))
            remaining_budget -= 1
        except Exception as exc:
            _log(f"[evaluate] !! reference lookup failed for {item.url}: {exc}")
            reference_cache[url_key] = max(assessment.reference_count, 0)
            remaining_budget -= 1

    for item, assessment, category in assessed:
        key = _normalize_title(item.title)
        url_key = _normalize_url(item.url)
        source_refs = title_groups.get(key, [item.source]) if key else [item.source]
        fallback_count = max(assessment.reference_count, len(source_refs))
        reference_count = reference_cache.get(url_key, fallback_count)
        trending_score = compute_trending_score(reference_count, item.published_at)

        trends.append(
            TrendItem(
                id=_hash_id(f"{item.source}:{item.title}"),
                category=category,
                title=item.title,
                announcement=item.source,
                url=item.url,
                published_at=item.published_at or datetime.utcnow(),
                source=item.source,
                summary=item.summary,
                impact_score=assessment.impact_score,
                reference_count=reference_count,
                trending_score=trending_score,
                source_references=source_refs,
            )
        )

    deduped: dict[str, TrendItem] = {}
    for trend in trends:
        key = _normalize_title(trend.title) or trend.id
        existing = deduped.get(key)
        if not existing or trend.trending_score > existing.trending_score:
            deduped[key] = trend

    sorted_trends = sorted(deduped.values(), key=lambda x: x.trending_score, reverse=True)
    per_category_counts = {"product": 0, "research": 0, "infra": 0}
    limited: List[TrendItem] = []
    for trend in sorted_trends:
        if per_category_counts.get(trend.category, 0) >= MAX_TRENDS_PER_CATEGORY:
            continue
        per_category_counts[trend.category] = per_category_counts.get(trend.category, 0) + 1
        limited.append(trend)

    _log(f"[evaluate] Done. Assessed trends: {len(limited)}")
    return state.model_copy(update={"assessed_items": limited})


def store_results(state: GraphState) -> GraphState:
    if state.assessed_items:
        products: dict[str, dict] = {}
        research: dict[str, dict] = {}
        infra: dict[str, dict] = {}

        def add_entry(target: dict, item: TrendItem) -> None:
            if len(target) >= MAX_TRENDS_PER_CATEGORY:
                return
            key = item.title
            if key in target:
                suffix = 2
                while f"{key} ({suffix})" in target:
                    suffix += 1
                key = f"{key} ({suffix})"
            target[key] = {
                "url": item.url,
                "publication": item.announcement,
                "publication_date": item.published_at.date().isoformat() if item.published_at else None,
                "trending_score": item.trending_score,
            }

        for item in sorted(state.assessed_items, key=lambda x: x.trending_score, reverse=True):
            if item.category == "product":
                add_entry(products, item)
            elif item.category == "infra":
                add_entry(infra, item)
            else:
                add_entry(research, item)

        run_date = date.fromisoformat(state.run_date) if state.run_date else datetime.utcnow().date()
        _log(
            f"[store] Upserting daily record for {run_date.isoformat()} "
            f"({len(products)} products, {len(research)} research, {len(infra)} infra)..."
        )
        upsert_daily_record(run_date, products, research, infra)
        _log("[store] Daily upsert complete.")
    else:
        print("[store] No assessed items; skipping Supabase upsert.", flush=True)
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("collect", collect_sources)
    graph.add_node("screen", screen_sources)
    graph.add_node("review", review_lookback)
    graph.add_node("evaluate", evaluate_sources)
    graph.add_node("store", store_results)

    graph.set_entry_point("collect")
    graph.add_edge("collect", "screen")
    graph.add_edge("screen", "review")
    graph.add_conditional_edges("review", should_expand, {"collect": "collect", "evaluate": "evaluate"})
    graph.add_edge("evaluate", "store")
    graph.set_finish_point("store")

    return graph


def run_daily(lookback_days: int | None = None) -> GraphState:
    run_date = datetime.utcnow().date()
    target_limit = max(TARGET_TRENDS, MAX_TRENDS_PER_CATEGORY * 3)
    exists = daily_record_exists(run_date)
    if not OVERWRITE_EXECUTION and exists:
        _log(f"[run] Daily record for {run_date.isoformat()} already exists. Skipping execution.")
        return GraphState(
            run_date=run_date.isoformat(),
            lookback_days=lookback_days or DEFAULT_LOOKBACK_DAYS,
            target_trend_count=target_limit,
            max_lookback_days=MAX_LOOKBACK_DAYS,
            errors=[f"Daily record exists for {run_date.isoformat()}"],
        )

    graph = build_graph().compile()
    state = GraphState(
        run_date=run_date.isoformat(),
        lookback_days=lookback_days or DEFAULT_LOOKBACK_DAYS,
        target_trend_count=target_limit,
        max_lookback_days=MAX_LOOKBACK_DAYS,
    )
    result = graph.invoke(state)
    #LangGraph returns a dict, convert it back to GraphState
    if isinstance(result, dict):
        return GraphState(**result)
    return result
