from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import List
from urllib.parse import urldefrag

from langgraph.graph import StateGraph

from .agents import evaluate_items, screen_items
from .config import (
    COMPUTE_TRENDING_SCORE,
    COLLECTION_LIMIT_PER_CATEGORY,
    DEFAULT_LOOKBACK_DAYS,
    MAX_COLLECTION_PASSES,
    MAX_REFERENCE_LOOKUPS,
    MAX_TRENDS_PER_CATEGORY,
    OVERWRITE_EXECUTION,
    TRENDS_MAX_WORKERS,
    TRENDS_VERBOSE,
)
from .curated_sources import FEED_SOURCES
from .schemas import GraphState, SourceItem, TrendItem
from .scoring import compute_trending_score
from .supabase_store import run_record_exists, upsert_run_record
from .tools import count_references, fetch_feed


def _log(message: str) -> None:
    if TRENDS_VERBOSE:
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


def _count_by_category(items: List[SourceItem]) -> dict[str, int]:
    counts = {"product": 0, "research": 0, "infra": 0}
    for item in items:
        if item.category:
            counts[item.category] = counts.get(item.category, 0) + 1
    return counts


def _round_robin_append(
    collected: List[SourceItem],
    counts: dict[str, int],
    feed: FeedSource,
    results: List[dict],
    seen_titles: set[str],
    seen_urls: set[str],
) -> None:
    for raw in results:
        if counts.get(feed.category, 0) >= COLLECTION_LIMIT_PER_CATEGORY:
            break
        payload = dict(raw)
        payload["category"] = feed.category
        item = SourceItem(**payload)
        title_key = _normalize_title(item.title)
        url_key = _normalize_url(item.url)
        if (title_key and title_key in seen_titles) or (url_key and url_key in seen_urls):
            continue
        if title_key:
            seen_titles.add(title_key)
        if url_key:
            seen_urls.add(url_key)
        collected.append(item)
        counts[feed.category] = counts.get(feed.category, 0) + 1
        break


def collect_sources(state: GraphState) -> GraphState:
    errors = list(state.errors)
    collected: List[SourceItem] = []
    inactive = set(state.inactive_categories)
    existing_kept = list(state.raw_items)
    existing_all = existing_kept + list(state.pending_items)
    counts = _count_by_category(existing_kept)
    seen_titles: set[str] = {_normalize_title(item.title) for item in existing_all if item.title}
    seen_urls: set[str] = {_normalize_url(item.url) for item in existing_all if item.url}
    attempted_categories: set[str] = set()

    _log(f"[collect] Starting curated feed collection ({len(FEED_SOURCES)} feeds)...")
    feed_entries: List[tuple[FeedSource, List[dict]]] = []
    for feed in FEED_SOURCES:
        attempted_categories.add(feed.category)
        if feed.category in inactive:
            continue
        if counts.get(feed.category, 0) >= COLLECTION_LIMIT_PER_CATEGORY:
            continue
        try:
            start = time.perf_counter()
            _log(f"[collect] -> fetch_feed {feed.name}")
            results = fetch_feed.invoke(
                {
                    "feed_url": feed.feed_url,
                    "lookback_days": state.lookback_days,
                    "source_name": feed.name,
                }
            )
            elapsed = time.perf_counter() - start
            _log(f"[collect] <- fetch_feed {feed.name} ({len(results)} items, {elapsed:.1f}s)")
        except Exception as exc:
            errors.append(f"fetch_feed/{feed.feed_url}: {exc}")
            _log(f"[collect] !! fetch_feed failed for {feed.feed_url}: {exc}")
            continue
        feed_entries.append((feed, results))

    progress = True
    while progress:
        progress = False
        for feed, results in feed_entries:
            if feed.category in inactive:
                continue
            if counts.get(feed.category, 0) >= COLLECTION_LIMIT_PER_CATEGORY:
                continue
            before = len(collected)
            _round_robin_append(collected, counts, feed, results, seen_titles, seen_urls)
            if len(collected) > before:
                progress = True

    total_new = len(collected)
    _log(
        "[collect] Done. New items: "
        f"{total_new} (product={counts['product']}, research={counts['research']}, infra={counts['infra']})"
    )
    return state.model_copy(
        update={
            "pending_items": collected,
            "errors": errors,
            "collection_pass": state.collection_pass + 1,
            "last_collect_added": total_new,
            "last_collect_categories": sorted(list(attempted_categories)),
        }
    )


def screen_sources(state: GraphState) -> GraphState:
    if not state.pending_items:
        _log("[screen] No new items to review.")
        return state
    _log(f"[screen] Reviewing {len(state.pending_items)} items for relevance...")
    screened = screen_items(state.pending_items)
    if not screened:
        _log("[screen] All new items discarded.")
        screened = []
    kept = list(state.raw_items) + screened
    counts = _count_by_category(kept)
    inactive = set(state.inactive_categories)
    for category in state.last_collect_categories:
        if counts.get(category, 0) == 0:
            inactive.add(category)
    _log(f"[screen] Kept {len(screened)} new items after screening.")
    return state.model_copy(
        update={
            "raw_items": kept,
            "pending_items": [],
            "inactive_categories": sorted(list(inactive)),
        }
    )


def should_collect_more(state: GraphState) -> str:
    if state.last_collect_added == 0:
        return "evaluate"
    if state.collection_pass >= MAX_COLLECTION_PASSES:
        return "evaluate"
    counts = _count_by_category(state.raw_items)
    inactive = set(state.inactive_categories)
    for category, count in counts.items():
        if category in inactive:
            continue
        if count < MAX_TRENDS_PER_CATEGORY:
            return "collect"
    return "evaluate"


def evaluate_sources(state: GraphState) -> GraphState:
    _log(
        f"[evaluate] Evaluating {len(state.raw_items)} items "
        f"(compute_trending_score={COMPUTE_TRENDING_SCORE})..."
    )
    assessed = evaluate_items(state.raw_items)
    trends: List[TrendItem] = []

    title_groups: dict[str, List[str]] = {}
    for item in state.raw_items:
        key = _normalize_title(item.title)
        title_groups.setdefault(key, []).append(item.source)

    reference_cache: dict[str, int] = {}
    reference_source: dict[str, str] = {}

    if COMPUTE_TRENDING_SCORE:
        prioritized = sorted(
            assessed,
            key=lambda x: x[0].published_at or datetime.min,
            reverse=True,
        )

        lookup_items: List[tuple[SourceItem, str]] = []
        for item, _assessment, _category in prioritized:
            if len(lookup_items) >= MAX_REFERENCE_LOOKUPS:
                break
            url_key = _normalize_url(item.url)
            if not url_key or url_key in reference_cache:
                continue
            lookup_items.append((item, url_key))

        if lookup_items:
            max_workers = min(TRENDS_MAX_WORKERS, 8, len(lookup_items))

            def lookup_reference(item: SourceItem) -> tuple[int, int, int]:
                payload = count_references.invoke(
                    {
                        "source_url": item.url,
                        "title": item.title,
                        "published_at": item.published_at.isoformat() if item.published_at else None,
                    }
                )
                return (
                    int(payload.get("coverage_count", 0)),
                    int(payload.get("url_count", 0)),
                    int(payload.get("title_count", 0)),
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(lookup_reference, item): (item, url_key)
                    for item, url_key in lookup_items
                }
                for future in as_completed(future_to_item):
                    item, url_key = future_to_item[future]
                    try:
                        coverage_count, url_count, title_count = future.result()
                        reference_cache[url_key] = coverage_count
                        if coverage_count == 0:
                            reference_source[url_key] = "lookup-empty"
                        elif coverage_count == url_count == title_count:
                            reference_source[url_key] = "lookup-url+title"
                        elif coverage_count == url_count:
                            reference_source[url_key] = "lookup-url"
                        else:
                            reference_source[url_key] = "lookup-title"
                    except Exception as exc:
                        _log(f"[evaluate] !! reference lookup failed for {item.url}: {exc}")
                        title_key = _normalize_title(item.title)
                        fallback_sources = title_groups.get(title_key, [item.source]) if title_key else [item.source]
                        reference_cache[url_key] = len(fallback_sources)
                        reference_source[url_key] = "fallback-title"
    else:
        _log("[evaluate] COMPUTE_TRENDING_SCORE=false; skipping reference lookups.")

    for item, assessment, category in assessed:
        key = _normalize_title(item.title)
        url_key = _normalize_url(item.url)
        source_refs = title_groups.get(key, [item.source]) if key else [item.source]
        fallback_count = len(source_refs)

        if COMPUTE_TRENDING_SCORE:
            if url_key and url_key in reference_cache:
                reference_count = reference_cache.get(url_key, fallback_count)
                ref_source = reference_source.get(url_key, "lookup")
            else:
                reference_count = fallback_count
                ref_source = "title-group"
            _log(f"[evaluate] references ({ref_source}) {reference_count} for {item.url}")
            trending_score = compute_trending_score(reference_count, item.published_at)
            if reference_count == 0:
                _log(f"[score] score 0 due to lack of references for {item.url}")
        else:
            # Use a lightweight proxy count (number of curated sources that surfaced the item)
            # but DO NOT persist trending_score downstream.
            reference_count = fallback_count
            trending_score = 0.0

        trends.append(
            TrendItem(
                id=_hash_id(f"{item.source}:{item.title}"),
                category=category,
                title=item.title,
                publication=item.source,
                url=item.url,
                published_at=item.published_at or datetime.utcnow(),
                source=item.source,
                summary=item.summary,
                reference_count=reference_count,
                trending_score=trending_score,
                source_references=source_refs,
            )
        )

    deduped: dict[str, TrendItem] = {}
    for trend in trends:
        key = _normalize_title(trend.title) or trend.id
        existing = deduped.get(key)
        if not existing:
            deduped[key] = trend
            continue

        if COMPUTE_TRENDING_SCORE:
            if trend.trending_score > existing.trending_score:
                deduped[key] = trend
        else:
            cand = (trend.published_at or datetime.min, trend.reference_count)
            prev = (existing.published_at or datetime.min, existing.reference_count)
            if cand > prev:
                deduped[key] = trend

    if COMPUTE_TRENDING_SCORE:
        sorted_trends = sorted(deduped.values(), key=lambda x: x.trending_score, reverse=True)
    else:
        sorted_trends = sorted(
            deduped.values(),
            key=lambda x: (x.published_at or datetime.min, x.reference_count),
            reverse=True,
        )

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
        trend_window = None

        def add_entry(target: dict, item: TrendItem) -> None:
            if len(target) >= MAX_TRENDS_PER_CATEGORY:
                return
            key = item.title
            if key in target:
                suffix = 2
                while f"{key} ({suffix})" in target:
                    suffix += 1
                key = f"{key} ({suffix})"

            entry = {
                "url": item.url,
                "publication": item.publication,
                "publication_date": item.published_at.date().isoformat() if item.published_at else None,
            }

            if COMPUTE_TRENDING_SCORE:
                entry["trending_score"] = item.trending_score

            target[key] = entry

        if COMPUTE_TRENDING_SCORE:
            ordered = sorted(state.assessed_items, key=lambda x: x.trending_score, reverse=True)
        else:
            ordered = sorted(
                state.assessed_items,
                key=lambda x: x.published_at or datetime.min,
                reverse=True,
            )

        for item in ordered:
            if item.category == "product":
                add_entry(products, item)
            elif item.category == "infra":
                add_entry(infra, item)
            else:
                add_entry(research, item)

        dates = [item.published_at.date() for item in state.assessed_items if item.published_at]
        if dates:
            start_date = min(dates)
            end_date = max(dates)
            trend_window = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }

        run_date = date.fromisoformat(state.run_date) if state.run_date else datetime.utcnow().date()
        _log(
            f"[store] Upserting run record for {run_date.isoformat()} "
            f"({len(products)} products, {len(research)} research, {len(infra)} infra)..."
        )
        upsert_run_record(run_date, products, research, infra, trend_window)
        _log("[store] Run upsert complete.")
    else:
        print("[store] No assessed items; skipping Supabase upsert.", flush=True)
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("collect", collect_sources)
    graph.add_node("screen", screen_sources)
    graph.add_node("evaluate", evaluate_sources)
    graph.add_node("store", store_results)

    graph.set_entry_point("collect")
    graph.add_edge("collect", "screen")
    graph.add_conditional_edges("screen", should_collect_more, {"collect": "collect", "evaluate": "evaluate"})
    graph.add_edge("evaluate", "store")
    graph.set_finish_point("store")

    return graph


def run(lookback_days: int | None = None) -> GraphState:
    run_date = datetime.utcnow().date()
    exists = run_record_exists(run_date)
    if not OVERWRITE_EXECUTION and exists:
        _log(f"[run] Run record for {run_date.isoformat()} already exists. Skipping execution.")
        return GraphState(
            run_date=run_date.isoformat(),
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            errors=[f"Run record exists for {run_date.isoformat()}"],
        )

    graph = build_graph().compile()
    state = GraphState(
        run_date=run_date.isoformat(),
        lookback_days=DEFAULT_LOOKBACK_DAYS,
    )
    result = graph.invoke(state)
    # LangGraph returns a dict, convert it back to GraphState
    if isinstance(result, dict):
        return GraphState(**result)
    return result
