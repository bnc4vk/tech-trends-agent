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
    print(f"[run] Done in {elapsed:.1f}s")
    if state.errors:
        print("[run] Errors:")
        for err in state.errors:
            print(f"- {err}")


if __name__ == "__main__":
    run_cli()
