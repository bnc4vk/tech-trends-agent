create table if not exists public.tech_trends (
  id text primary key,
  category text not null,
  title text not null,
  announcement text not null,
  url text not null,
  published_at timestamptz,
  source text,
  summary text,
  impact_score numeric,
  reference_count integer,
  trending_score numeric,
  source_references text[]
);

create index if not exists trends_category_idx on public.trends (category);
create index if not exists trends_trending_score_idx on public.trends (trending_score desc);
