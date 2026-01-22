from __future__ import annotations

from datetime import datetime
import math
from typing import Optional


def compute_trending_score(
    impact_score: float,
    reference_count: int,
    published_at: Optional[datetime],
) -> float:
    """Weighted score: impact (0-100), references, recency."""
    recency_bonus = 0.0
    if published_at:
        days_ago = max((datetime.utcnow() - published_at).days, 0)
        recency_bonus = max(0.0, 10.0 - days_ago * 1.5)

    reference_signal = math.log1p(max(reference_count, 0)) * 6.0
    impact_signal = max(min(impact_score, 100.0), 0.0) * 0.8
    return round(impact_signal + reference_signal + recency_bonus, 2)
