import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trends.run import run_cli  # noqa: E402


if __name__ == "__main__":
    run_cli()
