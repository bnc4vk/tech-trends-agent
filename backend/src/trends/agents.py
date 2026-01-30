from __future__ import annotations

import time
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

from .config import (
    MIN_SCREEN_CONFIDENCE,
    MISTRAL_API_KEY,
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
You are a strict tech-trend screener. Your job: decide whether the item is a meaningful tech trend signal.

KEEP=true only if the item indicates at least ONE of:
- A real product/feature launch or major update with clear technical substance
- A research/engineering breakthrough (paper, benchmark, method) with concrete claims
- An infrastructure/platform shift (standards, major cloud/platform changes, chips, frameworks)
- Clear industry adoption/usage signals (major deployment, rollout, integration, procurement)
- A significant security incident/vulnerability with broad relevance

DISCARD (KEEP=false) if it is primarily:
- Job posting / careers / hiring page / internships
- Event announcement without substantive technical content
- Link list, RSS directory, “roundup” with no new info
- Minor changelog, routine patch notes, trivial version bump
- Marketing fluff, vague thought leadership, sales pitch, spam, off-topic
- Duplicate/near-duplicate of widely repeated news with no new detail

If the summary/title are too vague to justify KEEP, default to KEEP=false.

Return ONLY valid JSON with exactly these keys:
{{
  "keep": boolean,
  "confidence": number,
  "rationale": string
}}

Confidence rubric:
- 0.85–1.00: strongly supported by specific facts (who/what/launched/results/dates)
- 0.60–0.84: likely trend, some specifics but not fully verified
- 0.35–0.59: ambiguous / partial signal / could be fluff
- 0.00–0.34: low-info, off-topic, spam, or clearly not a trend
""".strip(),
        ),
        (
            "user",
            """
Source: {source}
Title: {title}
Summary: {summary}
URL: {url}

Decide keep/confidence/rationale.
""".strip(),
        ),
    ]
)

def _log(message: str) -> None:
    if TRENDS_VERBOSE:
        print(message, flush=True)


def _build_llm() -> Optional[ChatOpenAI]:
    if not MISTRAL_API_KEY:
        return None
    return ChatMistralAI(
        model=TRENDS_LLM_MODEL,
        temperature=0.2,
        api_key=MISTRAL_API_KEY,
        timeout=TRENDS_LLM_TIMEOUT,
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
