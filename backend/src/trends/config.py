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
SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_SECRET_KEY = _env("SUPABASE_SECRET_KEY")
SUPABASE_TABLE = _env("SUPABASE_TABLE", "tech_trends")

TAVILY_API_KEY = _env("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = _env("BRAVE_SEARCH_API_KEY")
SERPAPI_API_KEY = _env("SERPAPI_API_KEY")

DEFAULT_LOOKBACK_DAYS = _env_int("DEFAULT_LOOKBACK_DAYS", 3)
TARGET_TRENDS = _env_int("TARGET_TRENDS", 20)
MAX_LOOKBACK_DAYS = _env_int("MAX_LOOKBACK_DAYS", 6)
LOOKBACK_STEP_DAYS = _env_int("LOOKBACK_STEP_DAYS", 3)
OVERWRITE_EXECUTION = _env_bool("OVERWRITE_EXECUTION", True)

SEARCH_PROVIDER = _env("SEARCH_PROVIDER", "tavily")
SEARCH_MAX_RESULTS = _env_int("SEARCH_MAX_RESULTS", 8)
SEARCH_MAX_QUERY_CHARS = _env_int("SEARCH_MAX_QUERY_CHARS", 380)
REFERENCE_SEARCH_MAX_RESULTS = _env_int("REFERENCE_SEARCH_MAX_RESULTS", 8)
REFERENCE_SEARCH_DEPTH = _env("REFERENCE_SEARCH_DEPTH", "basic")

MAX_SOURCES_PER_EXPERT = _env_int("MAX_SOURCES_PER_EXPERT", 6)
MAX_FEEDS_PER_SOURCE = _env_int("MAX_FEEDS_PER_SOURCE", 3)
MAX_ITEMS_PER_EXPERT = _env_int("MAX_ITEMS_PER_EXPERT", 30)
MAX_ITEMS_PER_SOURCE = _env_int("MAX_ITEMS_PER_SOURCE", 4)
MIN_UNIQUE_DOMAINS = _env_int("MIN_UNIQUE_DOMAINS", 4)
ALLOW_NON_RSS_SOURCES = _env_bool("ALLOW_NON_RSS_SOURCES", False)

MAX_TRENDS_PER_CATEGORY = _env_int("MAX_TRENDS_PER_CATEGORY", 12)
MAX_REFERENCE_LOOKUPS = _env_int("MAX_REFERENCE_LOOKUPS", 30)

TRENDS_VERBOSE = _env_bool("TRENDS_VERBOSE", True)
TRENDS_LLM_TIMEOUT = _env_float("TRENDS_LLM_TIMEOUT", 12.0)
TRENDS_MAX_WORKERS = _env_int("TRENDS_MAX_WORKERS", 12)
