from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


OPENAI_API_KEY = _env("OPENAI_API_KEY")
MISTRAL_API_KEY = _env("MISTRAL_API_KEY")
TRENDS_LLM_MODEL = _env("TRENDS_LLM_MODEL", "mistral-small-2506")
SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_SECRET_KEY = _env("SUPABASE_SECRET_KEY")
SUPABASE_TABLE = _env("SUPABASE_TABLE", "tech_trends")

TAVILY_API_KEY = _env("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = _env("BRAVE_SEARCH_API_KEY")
SERPAPI_API_KEY = _env("SERPAPI_API_KEY")

COMPUTE_TRENDING_SCORE = _env_bool("COMPUTE_TRENDING_SCORE", default=True)
DEFAULT_LOOKBACK_DAYS = _env_int("DEFAULT_LOOKBACK_DAYS", 3)
OVERWRITE_EXECUTION = _env_bool("OVERWRITE_EXECUTION", True)
COLLECTION_LIMIT_PER_CATEGORY = _env_int("COLLECTION_LIMIT_PER_CATEGORY", 18)
MAX_COLLECTION_PASSES = _env_int("MAX_COLLECTION_PASSES", 3)

SEARCH_PROVIDER = _env("SEARCH_PROVIDER", "tavily")
SEARCH_MAX_QUERY_CHARS = _env_int("SEARCH_MAX_QUERY_CHARS", 380)
REFERENCE_SEARCH_MAX_RESULTS = _env_int("REFERENCE_SEARCH_MAX_RESULTS", 8)
REFERENCE_SEARCH_DEPTH = _env("REFERENCE_SEARCH_DEPTH", "basic")

MAX_TRENDS_PER_CATEGORY = _env_int("MAX_TRENDS_PER_CATEGORY", 12)
MAX_REFERENCE_LOOKUPS = _env_int("MAX_REFERENCE_LOOKUPS", 30)
TREND_SCORE_HALF_LIFE_DAYS = _env_float("TREND_SCORE_HALF_LIFE_DAYS", 7.0)

TRENDS_VERBOSE = _env_bool("TRENDS_VERBOSE", True)
TRENDS_LLM_TIMEOUT = _env_float("TRENDS_LLM_TIMEOUT", 12.0)
TRENDS_MAX_WORKERS = _env_int("TRENDS_MAX_WORKERS", 12)
MIN_SCREEN_CONFIDENCE = _env_float("MIN_SCREEN_CONFIDENCE", 0.6)
