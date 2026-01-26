from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["product", "research", "infra"]


class SourceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[datetime] = None
    source: str
    summary: Optional[str] = None
    category: Optional[Category] = None


class SourceCandidate(BaseModel):
    title: str
    url: str


class TrendAssessment(BaseModel):
    category: Category = "product"


class TrendScreen(BaseModel):
    keep: bool
    rationale: str
    confidence: float = Field(ge=0, le=1)


class TrendItem(BaseModel):
    id: str
    category: Category
    title: str
    publication: str
    url: str
    published_at: Optional[datetime]
    source: str
    summary: Optional[str]
    reference_count: int
    trending_score: float
    source_references: List[str]


class GraphState(BaseModel):
    run_date: Optional[str] = None
    lookback_days: int = 3
    raw_items: List[SourceItem] = Field(default_factory=list)
    pending_items: List[SourceItem] = Field(default_factory=list)
    inactive_categories: List[Category] = Field(default_factory=list)
    collection_pass: int = 0
    last_collect_added: int = 0
    last_collect_categories: List[Category] = Field(default_factory=list)
    assessed_items: List[TrendItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
