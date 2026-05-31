from __future__ import annotations

import time

from .graph import run


def run_cli() -> None:
    started = time.perf_counter()
    print("[run] Starting trends pipeline...", flush=True)
    state = run()
    elapsed = time.perf_counter() - started
    print(f"[run] Collected {len(state.raw_items)} screened items")
    print(f"[run] Assessed {len(state.assessed_items)} trends")
    collection_metrics = state.metrics.get("collection", {})
    if collection_metrics:
        print(
            "[run] Collection metrics: "
            f"dynamic_discovery_share={collection_metrics.get('dynamic_discovery_share', 0):.2%}, "
            f"search_candidates={collection_metrics.get('total_search_candidates', 0)}, "
            f"total_candidates={collection_metrics.get('total_candidates', 0)}"
        )
    evaluation_metrics = state.metrics.get("evaluation", {})
    if evaluation_metrics:
        print(
            "[run] Evaluation metrics: "
            f"final_external_reference_coverage="
            f"{evaluation_metrics.get('final_external_reference_coverage', 0):.2%}, "
            f"distribution={evaluation_metrics.get('final_category_distribution', {})}"
        )
    print(f"[run] Done in {elapsed:.1f}s")
    if state.errors:
        print("[run] Errors:")
        for err in state.errors:
            print(f"- {err}")


if __name__ == "__main__":
    run_cli()
