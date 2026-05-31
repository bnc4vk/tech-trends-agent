from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trends import graph
from trends.schemas import GraphState, SourceItem, TrendAssessment


def _item(title: str, url: str, age_days: int, category: str = "product") -> SourceItem:
    return SourceItem(
        title=title,
        url=url,
        published_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=age_days),
        source=f"Source {url.rsplit('/', 1)[-1]}",
        summary="summary",
        category=category,
    )


def _assess(items: list[SourceItem]) -> list[tuple[SourceItem, TrendAssessment, str]]:
    return [(item, TrendAssessment(category=item.category or "product"), item.category or "product") for item in items]


class EvaluateSourcesRankingEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.patches = [
            patch.object(graph, "COMPUTE_TRENDING_SCORE", True),
            patch.object(graph, "TRENDS_VERBOSE", False),
            patch.object(graph, "TRENDS_MAX_WORKERS", 1),
            patch.object(graph, "evaluate_items", side_effect=_assess),
        ]
        for started in self.patches:
            started.start()
            self.addCleanup(started.stop)

    def _patch_lookup_counts(self, counts: dict[str, int]):
        class FakeCountReferences:
            @staticmethod
            def invoke(payload: dict) -> dict:
                coverage = counts[payload["source_url"]]
                return {
                    "coverage_count": coverage,
                    "url_count": coverage,
                    "title_count": coverage,
                }

        patcher = patch.object(graph, "count_references", FakeCountReferences)
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def test_final_trends_are_lookup_backed_when_category_has_enough_lookup_candidates(self) -> None:
        lookup_a = _item("Lookup A", "https://example.test/lookup-a", 0)
        lookup_b = _item("Lookup B", "https://example.test/lookup-b", 0)
        fallback_duplicates = [
            _item("Fallback Cluster", f"https://example.test/fallback-{index}", 1)
            for index in range(10)
        ]
        self._patch_lookup_counts({lookup_a.url: 1, lookup_b.url: 1})

        with patch.object(graph, "MAX_REFERENCE_LOOKUPS", 2), patch.object(graph, "MAX_TRENDS_PER_CATEGORY", 2):
            result = graph.evaluate_sources(GraphState(raw_items=[lookup_a, lookup_b, *fallback_duplicates]))

        titles = [item.title for item in result.assessed_items]
        self.assertEqual(titles, ["Lookup A", "Lookup B"])
        self.assertTrue(all(item.lookup_backed for item in result.assessed_items))
        self.assertEqual([item.reference_source for item in result.assessed_items], ["lookup-url+title", "lookup-url+title"])
        self.assertEqual(result.metrics["evaluation"]["final_lookup_coverage"], 1.0)
        self.assertEqual(result.metrics["evaluation"]["final_category_distribution"]["product"], 2)

    def test_final_trends_can_use_fallback_when_lookup_supply_is_short(self) -> None:
        lookup = _item("Lookup A", "https://example.test/lookup-a", 0)
        fallback_duplicates = [
            _item("Fallback Cluster", f"https://example.test/fallback-{index}", 1)
            for index in range(10)
        ]
        self._patch_lookup_counts({lookup.url: 1})

        with patch.object(graph, "MAX_REFERENCE_LOOKUPS", 1), patch.object(graph, "MAX_TRENDS_PER_CATEGORY", 2):
            result = graph.evaluate_sources(GraphState(raw_items=[lookup, *fallback_duplicates]))

        titles = {item.title for item in result.assessed_items}
        self.assertEqual(titles, {"Lookup A", "Fallback Cluster"})
        self.assertEqual(sum(1 for item in result.assessed_items if item.lookup_backed), 1)
        self.assertEqual(result.metrics["evaluation"]["final_lookup_backed_count"], 1)
        self.assertEqual(result.metrics["evaluation"]["final_lookup_coverage"], 0.5)
        self.assertEqual(result.metrics["evaluation"]["final_category_distribution"]["product"], 2)


if __name__ == "__main__":
    unittest.main()
