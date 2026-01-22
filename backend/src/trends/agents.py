from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY
from .schemas import Category, SourceItem, TrendAssessment


@dataclass
class DomainExpert:
    name: str
    category: Category
    description: str
    source_allowlist: Optional[List[str]] = None


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a domain expert evaluating a tech trend item for its likely impact.
Return a concise impact_score (0-100), reference_count estimate, and 1-2 sentence rationale.
""".strip(),
        ),
        (
            "user",
            """
Source: {source}
Title: {title}
Summary: {summary}
""".strip(),
        ),
    ]
)


KEYWORD_IMPACT = {
    "launch": 12,
    "release": 10,
    "open source": 8,
    "benchmark": 10,
    "paper": 8,
    "arxiv": 6,
    "model": 6,
    "inference": 6,
    "agent": 7,
    "changelog": 5,
    "summit": 5,
}


def _heuristic_assessment(item: SourceItem, category: Category) -> TrendAssessment:
    text = f"{item.title} {item.summary or ''}".lower()
    impact = 40.0
    for keyword, weight in KEYWORD_IMPACT.items():
        if keyword in text:
            impact += weight
    if category == "research":
        impact += 6
    impact = min(impact, 95.0)
    references = 1
    return TrendAssessment(
        impact_score=impact,
        reference_count=references,
        rationale="Heuristic scoring based on keyword density and category focus.",
    )


def _build_llm() -> Optional[ChatOpenAI]:
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)


def assess_item(item: SourceItem, category: Category, llm: Optional[ChatOpenAI]) -> TrendAssessment:
    if llm is None:
        return _heuristic_assessment(item, category)

    chain = PROMPT | llm.with_structured_output(TrendAssessment)
    return chain.invoke(
        {
            "source": item.source,
            "title": item.title,
            "summary": item.summary or "",
        }
    )


def route_category(item: SourceItem) -> Category:
    text = f"{item.title} {item.summary or ''}".lower()
    if any(token in text for token in ["paper", "arxiv", "benchmark", "dataset", "theorem"]):
        return "research"
    if item.source.lower() in [
        "arxiv sanity",
        "papers with code",
        "hugging face daily papers",
        "google deepmind blog",
    ]:
        return "research"
    return "product"


def build_experts() -> List[DomainExpert]:
    return [
        DomainExpert(
            name="Product Scout",
            category="product",
            description="Tracks product launches, changelogs, and platform upgrades.",
            source_allowlist=[
                "Ben's Bites",
                "GitHub Release Radar",
                "Vercel Changelog",
                "Deno Changelog",
                "Supabase Changelog",
                "OpenAI Blog",
                "Anthropic Blog",
            ],
        ),
        DomainExpert(
            name="Research Analyst",
            category="research",
            description="Evaluates papers, benchmarks, and model capability shifts.",
            source_allowlist=[
                "arXiv Sanity",
                "Papers with Code",
                "Hugging Face Daily Papers",
                "Google DeepMind Blog",
            ],
        ),
        DomainExpert(
            name="Infra Strategist",
            category="product",
            description="Assesses infra and compute landscape movements.",
            source_allowlist=["SemiAnalysis", "Latent Space", "Stratechery"],
        ),
    ]


def expert_can_handle(expert: DomainExpert, item: SourceItem) -> bool:
    if not expert.source_allowlist:
        return True
    return item.source in expert.source_allowlist


def evaluate_items(items: List[SourceItem]) -> List[tuple[SourceItem, TrendAssessment, Category]]:
    llm = _build_llm()
    experts = build_experts()
    assessed: List[tuple[SourceItem, TrendAssessment, Category]] = []

    for item in items:
        category = route_category(item)
        expert = next(
            (candidate for candidate in experts if candidate.category == category and expert_can_handle(candidate, item)),
            None,
        )
        assessment = assess_item(item, category, llm)
        assessed.append((item, assessment, category))
    return assessed
