import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trends.run_daily import run_daily  # noqa: E402


if __name__ == "__main__":
    started = time.perf_counter()
    print("[run] Starting daily trends pipeline...", flush=True)
    state = run_daily()
    elapsed = time.perf_counter() - started
    print(f"[run] Collected {len(state.raw_items)} raw items")
    print(f"[run] Assessed {len(state.assessed_items)} trends")
    print(f"[run] Done in {elapsed:.1f}s")
    if state.errors:
        print("[run] Errors:")
        for err in state.errors:
            print(f"- {err}")
