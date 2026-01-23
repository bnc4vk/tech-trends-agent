from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE")
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS"))
TARGET_TRENDS = int(os.getenv("TARGET_TRENDS"))
MAX_LOOKBACK_DAYS = int(os.getenv("MAX_LOOKBACK_DAYS"))
OVERWRITE_EXECUTION = int(os.getenv("OVERWRITE_EXECUTION"))
