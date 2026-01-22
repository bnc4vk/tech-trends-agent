# Tech Trends Agent (LangGraph)

This agentic pipeline collects daily tech trend signals, assigns expert evaluations, computes a trending score, and stores results in Supabase for the static front-end.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file using `.env.example` as a template.

## Supabase

Run the schema in `supabase.sql` in your Supabase SQL editor.

## Run the daily pipeline

```bash
python run_daily.py
```

This executes the LangGraph and upserts trend rows into Supabase.

## Customize sources

Update feed URLs or add new tools in `backend/src/trends/tools.py`.
