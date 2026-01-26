create table if not exists public.tech_trends (
  run_date date primary key,
  products jsonb not null,
  research jsonb not null,
  infra jsonb not null,
  trend_window jsonb,
  updated_at timestamptz default now()
);
