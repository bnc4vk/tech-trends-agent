from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY, TRENDS_LLM_TIMEOUT, TRENDS_MAX_WORKERS, TRENDS_VERBOSE
from .schemas import Category, SourceItem, TrendAssessment, TrendScreen


@dataclass
class DomainExpert:
    name: str
    category: Category
    description: str
    domain_description: str


SCREEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a general-purpose reader assessing whether an item represents a meaningful tech trend.
Keep items that describe real developments, launches, research advances, or infra shifts.
Discard items that are minor edits, change logs without substance, wiki/diff noise, spam, or off-topic posts.
Return a keep decision, confidence (0-1), and a brief rationale.
""".strip(),
        ),
        (
            "user",
            """
Source: {source}
Title: {title}
Summary: {summary}
URL: {url}
""".strip(),
        ),
    ]
)

def _log(message: str) -> None:
    if TRENDS_VERBOSE:
        print(message, flush=True)


RESEARCH_TOKENS = [
    "paper",
    "arxiv",
    "preprint",
    "benchmark",
    "dataset",
    "theorem",
    "sota",
    "state of the art",
    "ablation",
    "conference",
]

INFRA_TOKENS = [
    "gpu",
    "gpus",
    "cuda",
    "nvlink",
    "nccl",
    "tpu",
    "accelerator",
    "inference",
    "serving",
    "deployment",
    "kubernetes",
    "k8s",
    "datacenter",
    "data center",
    "cluster",
    "bandwidth",
    "throughput",
    "latency",
    "mlops",
    "infrastructure",
    "infra",
    "vector database",
    "vector db",
]

RESEARCH_SOURCE_TOKENS = [
    "arxiv sanity",
    "papers with code",
    "hugging face daily papers",
    "openreview",
    "deepmind blog",
]

INFRA_SOURCE_TOKENS = [
    "semianalysis",
    "nvidia",
    "amd",
    "intel",
    "coreweave",
    "lambda",
    "runpod",
    "together",
    "modal",
    "groq",
    "tenstorrent",
    "cerebras",
    "graphcore",
    "sambanova",
    "aws",
    "azure",
    "google cloud",
    "gcp",
]


def _build_llm() -> Optional[ChatOpenAI]:
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
        request_timeout=TRENDS_LLM_TIMEOUT,
        max_retries=1,
    )


def screen_item(item: SourceItem, llm: Optional[ChatOpenAI]) -> TrendScreen:
    if llm is None:
        return TrendScreen(keep=True, confidence=0.0, rationale="No LLM configured; default keep.")
    chain = SCREEN_PROMPT | llm.with_structured_output(TrendScreen)
    try:
        return chain.invoke(
            {
                "source": item.source,
                "title": item.title,
                "summary": item.summary or "",
                "url": item.url,
            }
        )
    except Exception as exc:
        return TrendScreen(keep=True, confidence=0.0, rationale=f"LLM screening error: {exc}")


def assess_item(item: SourceItem, category: Category) -> TrendAssessment:
    return TrendAssessment(
        category=category,
    )


def route_category(item: SourceItem) -> Category:
    text = f"{item.title} {item.summary or ''}".lower()
    source = item.source.lower()
    if any(token in text for token in RESEARCH_TOKENS) or any(token in source for token in RESEARCH_SOURCE_TOKENS):
        return "research"
    if any(token in text for token in INFRA_TOKENS) or any(token in source for token in INFRA_SOURCE_TOKENS):
        return "infra"
    return "product"


def build_experts() -> List[DomainExpert]:
    return [
        DomainExpert(
            name="Product Scout",
            category="product",
            description="Tracks product launches, changelogs, and platform upgrades.",
            domain_description=(
                "AI product launches, developer tools, platform releases, changelogs, "
                "official product blogs with RSS feeds, tech newsletters, API announcements"
            ),
        ),
        DomainExpert(
            name="Research Analyst",
            category="research",
            description="Evaluates papers, benchmarks, and model capability shifts.",
            domain_description=(
                "ML AI research papers, arXiv preprints, benchmarks, datasets, model releases, "
                "academic lab blogs with RSS feeds, research newsletters, conference proceedings"
            ),
        ),
        DomainExpert(
            name="Infra Strategist",
            category="infra",
            description="Assesses infra, compute, and deployment stack shifts.",
            domain_description=(
                "AI infrastructure, GPUs, accelerators, inference/serving stacks, model hosting, "
                "cloud compute, vector databases, MLOps, infra vendor blogs with RSS feeds"
            ),
        ),
    ]


def _evaluate_single_item(
    item: SourceItem, idx: int, total: int
) -> tuple[SourceItem, TrendAssessment, Category]:
    """Evaluate a single item - designed for parallel execution."""
    category_hint = route_category(item)
    start = time.perf_counter()
    if TRENDS_VERBOSE:
        _log(f"[evaluate] -> {idx}/{total} {item.source}")
    assessment = assess_item(item, category_hint)
    elapsed = time.perf_counter() - start
    if TRENDS_VERBOSE:
        _log(f"[evaluate] <- {idx}/{total} {item.source} ({elapsed:.1f}s)")
    category = assessment.category or category_hint
    return (item, assessment, category)


def evaluate_items(items: List[SourceItem]) -> List[tuple[SourceItem, TrendAssessment, Category]]:
    """Evaluate items in parallel using ThreadPoolExecutor."""
    if not items:
        return []
    
    total = len(items)
    # Pre-allocate list to maintain order, using Optional to allow None during construction
    assessed: List[Optional[tuple[SourceItem, TrendAssessment, Category]]] = [None] * total
    
    _log(f"[evaluate] Processing {total} items with {TRENDS_MAX_WORKERS} workers...")
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=TRENDS_MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(_evaluate_single_item, item, idx + 1, total): idx
            for idx, item in enumerate(items)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                assessed[idx] = result
                completed += 1
                if TRENDS_VERBOSE and completed % 10 == 0:
                    _log(f"[evaluate] Progress: {completed}/{total} ({completed*100//total}%)")
            except Exception as exc:
                # Fallback to base assessment on error
                item = items[idx]
                category = route_category(item)
                assessment = assess_item(item, category)
                assessed[idx] = (item, assessment, category)
                completed += 1
                _log(f"[evaluate] !! Error evaluating {item.source}: {exc}")
    
    elapsed = time.perf_counter() - start_time
    _log(f"[evaluate] Completed {total} items in {elapsed:.1f}s ({total/elapsed:.1f} items/sec)")
    
    # Filter out any None values (shouldn't happen, but safety check)
    result: List[tuple[SourceItem, TrendAssessment, Category]] = [
        r for r in assessed if r is not None
    ]
    return result


def screen_items(items: List[SourceItem]) -> List[SourceItem]:
    if not items:
        return []
    llm = _build_llm()
    if llm is None:
        return items
    total = len(items)
    screened: List[Optional[SourceItem]] = [None] * total
    _log(f"[screen] Screening {total} items with {TRENDS_MAX_WORKERS} workers...")
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=TRENDS_MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(screen_item, item, llm): idx for idx, item in enumerate(items)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            item = items[idx]
            try:
                decision = future.result()
                if decision.keep:
                    screened[idx] = item
                elif TRENDS_VERBOSE:
                    _log(f"[screen] Discarded: {item.source} - {item.title} ({decision.rationale})")
                completed += 1
            except Exception as exc:
                screened[idx] = item
                completed += 1
                _log(f"[screen] !! Error screening {item.source}: {exc}")
            if TRENDS_VERBOSE and completed % 10 == 0:
                _log(f"[screen] Progress: {completed}/{total} ({completed*100//total}%)")

    elapsed = time.perf_counter() - start_time
    kept = [item for item in screened if item is not None]
    _log(f"[screen] Kept {len(kept)}/{total} items in {elapsed:.1f}s")
    return kept
