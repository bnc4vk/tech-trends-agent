from __future__ import annotations

from typing import List

from supabase import create_client

from .config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_TABLE, SUPABASE_URL
from .schemas import TrendItem


def _serialize(item: TrendItem) -> dict:
    payload = item.model_dump()
    if item.published_at:
        payload["published_at"] = item.published_at.isoformat()
    return payload


def upsert_trends(items: List[TrendItem]) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase credentials are missing.")

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    payload = [_serialize(item) for item in items]
    if not payload:
        return

    client.table(SUPABASE_TABLE).upsert(payload).execute()
