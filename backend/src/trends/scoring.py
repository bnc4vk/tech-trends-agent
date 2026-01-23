from __future__ import annotations

from datetime import datetime
from typing import Optional


def compute_trending_score(reference_count: int, published_at: Optional[datetime]) -> float:
    """Reference velocity: references per day since publication."""
    if published_at:
        days_since = max((datetime.utcnow() - published_at).days, 0)
    else:
        days_since = 0
    days_since = max(days_since, 1)
    return round(max(reference_count, 0) / days_since, 4)
