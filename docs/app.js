(() => {
  const config = window.TRENDS_CONFIG || {};
  const productsList = document.getElementById("products-list");
  const researchList = document.getElementById("research-list");
  const infraList = document.getElementById("infra-list");
  const lastUpdated = document.getElementById("last-updated");
  const sourceCount = document.getElementById("source-count");
  const template = document.getElementById("trend-row-template");

  const formatDate = (value) => {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  const parseDate = (value) => {
    if (!value) return null;
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return null;
    return parsed;
  };

  const formatLongDate = (value) => {
    if (!value) return null;
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return null;
    return parsed.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric"
    });
  };

  const calcTrendWindow = (record) => {
    if (!record) return "—";
    const start = formatLongDate(record?.trend_window?.start_date);
    const end = formatLongDate(record?.trend_window?.end_date);
    if (start && end) {
      return `${start} and ${end}`;
    }
    const fallback = formatLongDate(record.run_date || record.updated_at);
    return fallback ? `Trending between ${fallback} and ${fallback}` : "—";
  };

  const createRow = (item) => {
    const fragment = template.content.cloneNode(true);
    const title = fragment.querySelector(".trend-title");
    const meta = fragment.querySelector(".trend-meta");
    const scoreContainer = fragment.querySelector(".trend-score");
    const score = fragment.querySelector(".trend-score .value");

    title.textContent = "";
    if (item.url) {
      const link = document.createElement("a");
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = item.title || "Untitled trend";
      title.appendChild(link);
    } else {
      title.textContent = item.title || "Untitled trend";
    }

    meta.textContent = `${item.publication || "Source"} · ${formatDate(item.publication_date)}`;

    if (Number.isFinite(item.trending_score)) {
      if (score) {
        score.textContent = Math.round(item.trending_score);
      }
    } else if (scoreContainer) {
      // If no score was computed/persisted, hide the entire score pill/column.
      scoreContainer.remove();
    }

    return fragment;
  };

  const renderList = (items, container) => {
    container.innerHTML = "";
    if (!items.length) {
      container.innerHTML =
        '<p class="trend-meta">No trends yet — run the daily agent to populate data.</p>';
      return;
    }
    items.forEach((item) => container.appendChild(createRow(item)));
  };

  const fetchSupabaseTrends = async () => {
    if (!config.supabaseUrl || !config.supabaseAnonKey) {
      throw new Error("Missing Supabase configuration: supabaseUrl and supabaseAnonKey are required");
    }

    const query = new URLSearchParams({
      select: "*",
      order: "run_date.desc",
      limit: 1
    });

    const response = await fetch(
      `${config.supabaseUrl}/rest/v1/${config.tableName || "trends"}?${query}`,
      {
        headers: {
          apikey: config.supabaseAnonKey,
          Authorization: `Bearer ${config.supabaseAnonKey}`
        }
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch trends from Supabase: ${response.status} ${response.statusText}`);
    }

    const records = await response.json();
    if (!Array.isArray(records)) {
      throw new Error("Invalid data format: expected an array from Supabase");
    }

    return records[0] || null;
  };

  const toTrendArray = (bucket) => {
    if (!bucket || typeof bucket !== "object") return [];
    return Object.entries(bucket).map(([title, details]) => ({
      title,
      ...details
    }));
  };

  const sortTrends = (a, b) => {
    const aHasScore = Number.isFinite(a.trending_score);
    const bHasScore = Number.isFinite(b.trending_score);
    if (aHasScore && bHasScore) {
      return b.trending_score - a.trending_score;
    }
    if (aHasScore !== bHasScore) {
      // Prefer items that actually have a computed score.
      return aHasScore ? -1 : 1;
    }
    const aDate = parseDate(a.publication_date) || parseDate(a.run_date) || parseDate(a.updated_at);
    const bDate = parseDate(b.publication_date) || parseDate(b.run_date) || parseDate(b.updated_at);
    const aTime = aDate ? aDate.getTime() : 0;
    const bTime = bDate ? bDate.getTime() : 0;
    return bTime - aTime;
  };

  const init = async () => {
    let record;
    try {
      record = await fetchSupabaseTrends();
    } catch (error) {
      console.error("Error fetching trends:", error);
      record = null;
    }

    if (!record && config.fallbackDataPath) {
      const fallbackResponse = await fetch(config.fallbackDataPath);
      const fallback = await fallbackResponse.json();
      record = Array.isArray(fallback) ? fallback[0] : fallback;
    }

    const products = toTrendArray(record?.products).sort(sortTrends);
    const research = toTrendArray(record?.research).sort(sortTrends);
    const infra = toTrendArray(record?.infra).sort(sortTrends);

    renderList(products, productsList);
    renderList(research, researchList);
    renderList(infra, infraList);

    lastUpdated.textContent = calcTrendWindow(record);
    const publications = new Set(
      [...products, ...research, ...infra].map((item) => item.publication).filter(Boolean)
    );
    sourceCount.textContent = publications.size || "—";
  };

  init();
})();
