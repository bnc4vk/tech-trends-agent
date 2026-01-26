from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from .config import TREND_SCORE_HALF_LIFE_DAYS

def compute_trending_score(reference_count: int, published_at: Optional[datetime]) -> float:
    """Weighted coverage: log-scaled references with exponential decay by age."""
    references = max(reference_count, 0)
    if published_at:
        elapsed = datetime.utcnow() - published_at
        days_since = max(elapsed.total_seconds() / 86400, 0.0)
    else:
        days_since = 0.0
    half_life = max(TREND_SCORE_HALF_LIFE_DAYS, 0.1)
    score = math.log1p(references) * math.exp(-days_since / half_life)
    return round(score, 4)
