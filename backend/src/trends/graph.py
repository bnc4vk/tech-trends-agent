from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, date
from typing import List

from langgraph.graph import StateGraph

from .agents import build_experts, evaluate_items
from .config import DEFAULT_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS, OVERWRITE_EXECUTION, TARGET_TRENDS
from .schemas import GraphState, SourceItem, TrendItem
from .scoring import compute_trending_score
from .supabase_store import daily_record_exists, upsert_daily_record
from .tools import discover_feeds, fetch_feed, search_sources


VERBOSE = os.getenv("TRENDS_VERBOSE")
LOOKBACK_STEP_DAYS = int(os.getenv("LOOKBACK_STEP_DAYS"))
MAX_SOURCES_PER_EXPERT = int(os.getenv("MAX_SOURCES_PER_EXPERT"))
MAX_FEEDS_PER_SOURCE = int(os.getenv("MAX_FEEDS_PER_SOURCE"))

def _log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


def _normalize_title(title: str) -> str:
    return "".join(ch for ch in title.lower() if ch.isalnum() or ch.isspace()).strip()


def _hash_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def collect_sources(state: GraphState) -> GraphState:
    items: List[SourceItem] = []
    errors = list(state.errors)
    experts = build_experts()
    feed_urls: set[str] = set()

    _log(f"[collect] Starting discovery with {len(experts)} experts...")
    for expert in experts:
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
                        items.append(SourceItem(**raw))
                except Exception as exc:
                    errors.append(f"fetch_feed/{feed_url}: {exc}")
                    _log(f"[collect] !! fetch_feed failed for {feed_url}: {exc}")
            
            items_added = len(items) - item_count_before
            if feed_count > 0 and items_added == 0:
                _log(f"[collect] Found {feed_count} feed(s) for {source_url} but 0 items within {state.lookback_days} day lookback")

    _log(f"[collect] Done. Total items: {len(items)}")
    return state.model_copy(update={"raw_items": items, "errors": errors})


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

    title_groups: dict[str, List[SourceItem]] = {}
    for item in state.raw_items:
        key = _normalize_title(item.title)
        title_groups.setdefault(key, []).append(item)

    for item, assessment, category in assessed:
        key = _normalize_title(item.title)
        reference_count = max(assessment.reference_count, len(title_groups.get(key, [])))
        trending_score = compute_trending_score(
            assessment.impact_score,
            reference_count,
            item.published_at,
        )

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
                source_references=[source.source for source in title_groups.get(key, [])],
            )
        )

    _log(f"[evaluate] Done. Assessed trends: {len(trends)}")
    return state.model_copy(update={"assessed_items": trends})


def store_results(state: GraphState) -> GraphState:
    if state.assessed_items:
        products: dict[str, dict] = {}
        research: dict[str, dict] = {}
        infra: dict[str, dict] = {}

        def add_entry(target: dict, item: TrendItem) -> None:
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
    graph.add_node("review", review_lookback)
    graph.add_node("evaluate", evaluate_sources)
    graph.add_node("store", store_results)

    graph.set_entry_point("collect")
    graph.add_edge("collect", "review")
    graph.add_conditional_edges("review", should_expand, {"collect": "collect", "evaluate": "evaluate"})
    graph.add_edge("evaluate", "store")
    graph.set_finish_point("store")

    return graph


def run_daily(lookback_days: int | None = None) -> GraphState:
    run_date = datetime.utcnow().date()
    exists = daily_record_exists(run_date)
    if not OVERWRITE_EXECUTION and exists:
        _log(f"[run] Daily record for {run_date.isoformat()} already exists. Skipping execution.")
        return GraphState(
            run_date=run_date.isoformat(),
            lookback_days=lookback_days or DEFAULT_LOOKBACK_DAYS,
            target_trend_count=TARGET_TRENDS,
            max_lookback_days=MAX_LOOKBACK_DAYS,
            errors=[f"Daily record exists for {run_date.isoformat()}"],
        )

    graph = build_graph().compile()
    state = GraphState(
        run_date=run_date.isoformat(),
        lookback_days=lookback_days or DEFAULT_LOOKBACK_DAYS,
        target_trend_count=TARGET_TRENDS,
        max_lookback_days=MAX_LOOKBACK_DAYS,
    )
    result = graph.invoke(state)
    #LangGraph returns a dict, convert it back to GraphState
    if isinstance(result, dict):
        return GraphState(**result)
    return result
