import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trends.run_daily import run_daily  # noqa: E402


if __name__ == "__main__":
    state = run_daily()
    print(f"Collected {len(state.raw_items)} raw items")
    print(f"Assessed {len(state.assessed_items)} trends")
    if state.errors:
        print("Errors:")
        for err in state.errors:
            print(f"- {err}")
