from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["product", "research"]


class SourceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[datetime] = None
    source: str
    summary: Optional[str] = None


class TrendAssessment(BaseModel):
    impact_score: float = Field(ge=0, le=100)
    reference_count: int = Field(ge=0)
    rationale: str


class TrendItem(BaseModel):
    id: str
    category: Category
    title: str
    announcement: str
    url: str
    published_at: Optional[datetime]
    source: str
    summary: Optional[str]
    impact_score: float
    reference_count: int
    trending_score: float
    references: List[str]


class GraphState(BaseModel):
    lookback_days: int = 2
    raw_items: List[SourceItem] = Field(default_factory=list)
    assessed_items: List[TrendItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
