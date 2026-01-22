from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from typing import List

from langgraph.graph import StateGraph

from .agents import evaluate_items
from .config import DEFAULT_LOOKBACK_DAYS
from .schemas import GraphState, SourceItem, TrendItem
from .scoring import compute_trending_score
from .supabase_store import upsert_trends
from .tools import TREND_TOOLS


VERBOSE = os.getenv("TRENDS_VERBOSE", "1") not in {"0", "false", "False"}


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
    _log(f"[collect] Starting collection from {len(TREND_TOOLS)} sources...")
    for tool in TREND_TOOLS:
        try:
            start = time.perf_counter()
            _log(f"[collect] -> {tool.name}")
            results = tool.invoke({"lookback_days": state.lookback_days})
            for raw in results:
                items.append(SourceItem(**raw))
            elapsed = time.perf_counter() - start
            _log(f"[collect] <- {tool.name} ({len(results)} items, {elapsed:.1f}s)")
        except Exception as exc:
            errors.append(f"{tool.name}: {exc}")
            _log(f"[collect] !! {tool.name} failed: {exc}")
    _log(f"[collect] Done. Total items: {len(items)}")
    return state.model_copy(update={"raw_items": items, "errors": errors})


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
        _log(f"[store] Upserting {len(state.assessed_items)} trends to Supabase...")
        upsert_trends(state.assessed_items)
        _log("[store] Upsert complete.")
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("collect", collect_sources)
    graph.add_node("evaluate", evaluate_sources)
    graph.add_node("store", store_results)

    graph.set_entry_point("collect")
    graph.add_edge("collect", "evaluate")
    graph.add_edge("evaluate", "store")
    graph.set_finish_point("store")

    return graph


def run_daily(lookback_days: int | None = None) -> GraphState:
    graph = build_graph().compile()
    state = GraphState(lookback_days=lookback_days or DEFAULT_LOOKBACK_DAYS)
    result = graph.invoke(state)
    # LangGraph returns a dict, convert it back to GraphState
    if isinstance(result, dict):
        return GraphState(**result)
    return result
