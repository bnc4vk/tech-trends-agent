(() => {
  const config = window.TRENDS_CONFIG || {};
  const productsList = document.getElementById("products-list");
  const researchList = document.getElementById("research-list");
  const lastUpdated = document.getElementById("last-updated");
  const sourceCount = document.getElementById("source-count");
  const template = document.getElementById("trend-row-template");

  const formatDate = (value) => {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  const calcLastUpdated = (items) => {
    if (!items.length) return "—";
    const maxDate = items
      .map((item) => new Date(item.published_at || item.updated_at || item.created_at))
      .filter((d) => !Number.isNaN(d.getTime()))
      .sort((a, b) => b - a)[0];
    if (!maxDate) return "—";
    return maxDate.toLocaleDateString(undefined, {
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
      ? `<a href="${item.url}" target="_blank" rel="noreferrer">${item.announcement || "Source"}</a>`
      : item.announcement || "Source";

    const refs = typeof item.reference_count === "number" ? item.reference_count : "—";
    meta.innerHTML = `${link} · ${formatDate(item.published_at)} · ${refs} refs`;

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
      order: "trending_score.desc",
      limit: config.maxRows || 20
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

    const items = await response.json();
    if (!Array.isArray(items)) {
      throw new Error("Invalid data format: expected an array from Supabase");
    }

    return items;
  };

  const init = async () => {
    let items;
    try {
      items = await fetchSupabaseTrends();
    } catch (error) {
      console.error("Error fetching trends:", error);
      throw error;
    }

    items = items.sort((a, b) => (b.trending_score || 0) - (a.trending_score || 0));

    const products = items.filter((item) => item.category === "product");
    const research = items.filter((item) => item.category === "research");

    renderList(products, productsList);
    renderList(research, researchList);

    lastUpdated.textContent = calcLastUpdated(items);
    sourceCount.textContent = new Set(items.map((item) => item.announcement)).size || "—";
  };

  init();
})();
