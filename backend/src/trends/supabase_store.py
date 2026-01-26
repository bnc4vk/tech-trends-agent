from __future__ import annotations

from datetime import date, datetime

from supabase import create_client

from .config import SUPABASE_SECRET_KEY, SUPABASE_TABLE, SUPABASE_URL


def daily_record_exists(run_date: date) -> bool:
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError("Supabase credentials are missing.")

    client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    response = client.table(SUPABASE_TABLE).select("run_date").eq("run_date", run_date.isoformat()).execute()
    
    return bool(response.data)


def upsert_daily_record(
    run_date: date,
    products: dict,
    research: dict,
    infra: dict,
    trend_window: dict | None = None,
) -> None:
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError("Supabase credentials are missing.")

    client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    payload = {
        "run_date": run_date.isoformat(),
        "products": products,
        "research": research,
        "infra": infra,
        "trend_window": trend_window,
        "updated_at": datetime.utcnow().isoformat(),
    }
    client.table(SUPABASE_TABLE).upsert(payload).execute()
