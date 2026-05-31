from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trends import graph  # noqa: E402
from trends import tools  # noqa: E402
from trends.curated_sources import FeedSource  # noqa: E402
from trends.schemas import GraphState, SourceCandidate, SourceItem  # noqa: E402


def _feed_items(category: str, count: int, prefix: str) -> list[dict]:
    now = datetime.utcnow()
    return [
        SourceItem(
            title=f"{prefix} feed story {idx}",
            url=f"https://feeds.example/{category}/{idx}",
            published_at=now - timedelta(minutes=idx),
            source=f"{prefix} Feed",
            category=category,
        ).model_dump()
        for idx in range(count)
    ]


def _search_items(category: str, count: int) -> list[dict]:
    now = datetime.utcnow()
    return [
        SourceItem(
            title=f"{category} search story {idx}",
            url=f"https://search.example/{category}/{idx}",
            published_at=now - timedelta(minutes=idx),
            source="Dynamic Search (test)",
            category=category,
            discovery_method="search",
            discovery_query=f"{category} query",
        ).model_dump()
        for idx in range(count)
    ]


class StrategyMetricsTests(unittest.TestCase):
    def test_dynamic_discovery_excludes_low_quality_social_domains(self) -> None:
        candidates = [
            SourceCandidate(title="Social post", url="https://www.facebook.com/group/posts/1"),
            SourceCandidate(title="Forum post", url="https://reddit.com/r/ml/comments/1"),
            SourceCandidate(title="Vendor launch", url="https://example.com/vendor-launch"),
        ]

        with (
            patch.object(tools, "DYNAMIC_DISCOVERY_ENABLED", True),
            patch.object(tools, "TAVILY_API_KEY", "test-key"),
            patch.object(tools, "_search_candidates", return_value=candidates),
        ):
            result = tools._discover_trending_candidates("product", lookback_days=3, max_results=8)

        self.assertEqual([item.url for item in result.items], ["https://example.com/vendor-launch"])
        self.assertEqual(result.metadata.dynamic_discovery_count, 1)

    def test_collect_reserves_measurable_dynamic_discovery_share(self) -> None:
        feeds = [
            FeedSource("product", "Product", "https://feeds.example/product"),
            FeedSource("research", "Research", "https://feeds.example/research"),
            FeedSource("infra", "Infra", "https://feeds.example/infra"),
        ]

        def fetch_side_effect(payload: dict) -> list[dict]:
            category = payload["source_name"].lower()
            return _feed_items(category, 18, payload["source_name"])

        def search_side_effect(payload: dict) -> list[dict]:
            return _search_items(payload["category"], 8)

        with (
            patch.object(graph, "FEED_SOURCES", feeds),
            patch.object(graph, "COLLECTION_LIMIT_PER_CATEGORY", 18),
            patch.object(graph, "DYNAMIC_DISCOVERY_ENABLED", True),
            patch.object(graph, "DYNAMIC_DISCOVERY_PER_CATEGORY", 4),
            patch.object(graph, "fetch_feed", SimpleNamespace(invoke=fetch_side_effect)),
            patch.object(graph, "discover_trending_candidates", SimpleNamespace(invoke=search_side_effect)),
        ):
            state = graph.collect_sources(GraphState(lookback_days=3))

        metrics = state.metrics["collection"]
        self.assertEqual(len(state.pending_items), 54)
        self.assertEqual(metrics["new_feed_candidates"], 42)
        self.assertEqual(metrics["new_search_candidates"], 12)
        self.assertEqual(metrics["candidates_by_category"], {"product": 18, "research": 18, "infra": 18})
        self.assertAlmostEqual(metrics["dynamic_discovery_share"], 12 / 54, places=4)

    def test_evaluate_prefers_lookup_backed_final_distribution(self) -> None:
        now = datetime.utcnow()
        raw_items: list[SourceItem] = []
        for category in ("product", "research", "infra"):
            for idx in range(18):
                raw_items.append(
                    SourceItem(
                        title=f"{category} story {idx}",
                        url=f"https://articles.example/{category}/{idx}",
                        published_at=now - timedelta(minutes=idx),
                        source=f"{category} source",
                        category=category,
                    )
                )

        lookup_calls: list[str] = []

        def lookup_side_effect(payload: dict) -> dict:
            lookup_calls.append(payload["source_url"])
            return {
                "coverage_count": 10,
                "url_count": 2,
                "title_count": 10,
                "result_count": 2,
                "url_query": "",
                "title_query": "",
            }

        with (
            patch.object(graph, "COMPUTE_TRENDING_SCORE", True),
            patch.object(graph, "MAX_REFERENCE_LOOKUPS", 36),
            patch.object(graph, "MAX_TRENDS_PER_CATEGORY", 12),
            patch.object(graph, "TRENDS_MAX_WORKERS", 1),
            patch.object(graph, "count_references", SimpleNamespace(invoke=lookup_side_effect)),
        ):
            state = graph.evaluate_sources(GraphState(raw_items=raw_items))

        metrics = state.metrics["evaluation"]
        self.assertEqual(len(lookup_calls), 36)
        self.assertEqual(len(state.assessed_items), 36)
        self.assertEqual(metrics["final_category_distribution"], {"product": 12, "research": 12, "infra": 12})
        self.assertEqual(metrics["final_lookup_backed_count"], 36)
        self.assertEqual(metrics["final_external_reference_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
