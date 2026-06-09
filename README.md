# Agentic Tech Trends Curation

This repo contains a static GitHub Pages front-end and a LangGraph-based back-end pipeline that curates daily tech trends and computes a trending score.

## Front-end (GitHub Pages)

The static site lives in `docs/` and can be published via GitHub Pages.

1. Update `docs/config.js` with your Supabase URL and anon key.
2. Update the repo link in `docs/index.html`.
3. Point GitHub Pages at the `/docs` folder.

## Back-end (LangGraph)

See `backend/README.md` for setup instructions.

## Scheduled workflows

The weekly GitHub Actions pipelines update `.github/heartbeat/*.json` after successful runs and commit the heartbeat back to the active branch. This keeps repository activity visible while the GitHub Pages front-end remains in active use.

## Data flow

1. LangGraph agents collect and assess trends.
2. Results are stored in Supabase.
3. The static front-end fetches data from Supabase and renders the three-column layout.
