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

  const calcLastUpdated = (record) => {
    if (!record) return "—";
    const dateValue = record.run_date || record.updated_at;
    if (!dateValue) return "—";
    const parsed = new Date(dateValue);
    if (Number.isNaN(parsed.getTime())) return dateValue;
    return parsed.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric"
    });
  };

  const createRow = (item) => {
    const fragment = template.content.cloneNode(true);
    const title = fragment.querySelector(".trend-title");
    const meta = fragment.querySelector(".trend-meta");
    const score = fragment.querySelector(".trend-score .value");

    title.textContent = item.title || "Untitled trend";

    const link = item.url
      ? `<a href="${item.url}" target="_blank" rel="noreferrer">${item.publication || "Source"}</a>`
      : item.publication || "Source";

    meta.innerHTML = `${link} · ${formatDate(item.publication_date)}`;

    score.textContent = Number.isFinite(item.trending_score)
      ? Math.round(item.trending_score)
      : "—";

    return fragment;
  };

  const renderList = (items, container) => {
    container.innerHTML = "";
    if (!items.length) {
      container.innerHTML = "<p class=\"trend-meta\">No trends yet — run the daily agent to populate data.</p>";
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

    const products = toTrendArray(record?.products).sort(
      (a, b) => (b.trending_score || 0) - (a.trending_score || 0)
    );
    const research = toTrendArray(record?.research).sort(
      (a, b) => (b.trending_score || 0) - (a.trending_score || 0)
    );
    const infra = toTrendArray(record?.infra).sort(
      (a, b) => (b.trending_score || 0) - (a.trending_score || 0)
    );

    renderList(products, productsList);
    renderList(research, researchList);
    renderList(infra, infraList);

    lastUpdated.textContent = calcLastUpdated(record);
    const publications = new Set(
      [...products, ...research, ...infra].map((item) => item.publication).filter(Boolean)
    );
    sourceCount.textContent = publications.size || "—";
  };

  init();
})();
