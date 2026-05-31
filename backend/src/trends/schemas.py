from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["product", "research", "infra"]
DiscoveryMethod = Literal["feed", "search"]


class SourceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[datetime] = None
    source: str
    summary: Optional[str] = None
    category: Optional[Category] = None
    discovery_method: DiscoveryMethod = "feed"
    discovery_query: Optional[str] = None
    discovery_provider: Optional[str] = None
    discovery_rank: Optional[int] = Field(default=None, ge=1)


class SourceCandidate(BaseModel):
    title: str
    url: str
    summary: Optional[str] = None
    published_at: Optional[datetime] = None
    source: Optional[str] = None
    category: Optional[Category] = None
    discovery_method: DiscoveryMethod = "search"
    discovery_query: Optional[str] = None
    discovery_provider: Optional[str] = None
    discovery_rank: Optional[int] = Field(default=None, ge=1)


class DynamicDiscoveryMetadata(BaseModel):
    enabled: bool
    provider: str
    requested_queries: int = 0
    executed_queries: int = 0
    result_count: int = 0
    dynamic_discovery_count: int = 0
    max_queries: int = 0
    max_results_per_query: int = 0
    max_total_results: int = 0
    skipped_reason: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class DynamicDiscoveryResult(BaseModel):
    items: List[SourceItem] = Field(default_factory=list)
    metadata: DynamicDiscoveryMetadata


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
    reference_source: str = "title-group"
    lookup_backed: bool = False
    discovery_method: DiscoveryMethod = "feed"
    discovery_query: Optional[str] = None


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
    metrics: dict[str, Any] = Field(default_factory=dict)
