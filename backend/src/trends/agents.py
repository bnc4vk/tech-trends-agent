from __future__ import annotations

import time
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import (
    MIN_SCREEN_CONFIDENCE,
    OPENAI_API_KEY,
    TRENDS_LLM_MODEL,
    TRENDS_LLM_TIMEOUT,
    TRENDS_VERBOSE,
)
from .schemas import Category, SourceItem, TrendAssessment, TrendScreen


SCREEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict reviewer assessing whether an item represents a meaningful tech trend.
Keep items only if they describe real developments, notable launches, research advances,
infrastructure shifts, or clear industry adoption signals.
Discard items that are job postings, hiring/careers pages, internships, events without
substantive tech content, link lists or RSS directories, minor changelogs, marketing fluff,
spam, or anything off-topic.
If the item is not clearly a tech trend, set keep=false and use a low confidence score.
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


def _build_llm() -> Optional[ChatOpenAI]:
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(
        model=TRENDS_LLM_MODEL,
        temperature=0.2,
        api_key=OPENAI_API_KEY,
        request_timeout=TRENDS_LLM_TIMEOUT,
        max_retries=1,
    )


def _build_screen_chain(llm: ChatOpenAI):
    return SCREEN_PROMPT | llm.with_structured_output(TrendScreen)


def screen_item(item: SourceItem, chain) -> TrendScreen:
    if chain is None:
        return TrendScreen(keep=True, confidence=0.0, rationale="No LLM configured; default keep.")
    try:
        time.sleep(20)
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


def evaluate_items(items: List[SourceItem]) -> List[tuple[SourceItem, TrendAssessment, Category]]:
    if not items:
        return []
    assessed: List[tuple[SourceItem, TrendAssessment, Category]] = []
    for item in items:
        category = item.category or "product"
        assessment = TrendAssessment(category=category)
        assessed.append((item, assessment, category))
    return assessed


def screen_items(items: List[SourceItem]) -> List[SourceItem]:
    if not items:
        return []
    llm = _build_llm()
    if llm is None:
        return items
    chain = _build_screen_chain(llm)
    total = len(items)
    screened: List[SourceItem] = []
    _log(f"[screen] Screening {total} items with 20s interval...")
    start_time = time.perf_counter()
    completed = 0

    for item in items:
        decision = screen_item(item, chain)
        if decision.keep and decision.confidence >= MIN_SCREEN_CONFIDENCE:
            screened.append(item)
        elif TRENDS_VERBOSE:
            reason = decision.rationale
            if decision.keep and decision.confidence < MIN_SCREEN_CONFIDENCE:
                reason = f"low confidence: {reason}"
            _log(
                f"[screen] Discarded: {item.source} - {item.title} "
                f"(confidence={decision.confidence:.2f}, {reason})"
            )
        completed += 1
        if TRENDS_VERBOSE and completed % 5 == 0:
            _log(f"[screen] Progress: {completed}/{total} ({completed*100//total}%)")

    elapsed = time.perf_counter() - start_time
    _log(f"[screen] Kept {len(screened)}/{total} items in {elapsed:.1f}s")
    return screened
