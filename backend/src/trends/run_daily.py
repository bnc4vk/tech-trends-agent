from __future__ import annotations

from .graph import run_daily


if __name__ == "__main__":
    state = run_daily()
    print(f"Collected {len(state.raw_items)} raw items")
    print(f"Assessed {len(state.assessed_items)} trends")
    if state.errors:
        print("Errors:")
        for err in state.errors:
            print(f"- {err}")
