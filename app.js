"use strict";

/*
 * Sector-Relative Valuation - browser app.
 *
 * Ports src/dashboard.py to Plotly.js. Three CSVs feed the page:
 *   /sector_analysis.csv         per-ticker composites + PE + composite_z_score
 *   /sector_analysis_full.csv    per-ticker raw metrics + every z-score
 *   /weights.csv                 per-sector Ridge weights + alpha + r_squared
 *
 * Tabs lazy-load: each tab fetches the CSVs it needs the first time the
 * user activates it. Tab 1 (Sector Analysis) loads on initial paint so
 * the default view is populated before the user clicks anything.
 */

const COLORS = {
  background: "#f8f9fa",
  text: "#2c3e50",
  primary: "#3498db",
  secondary: "#e74c3c",
  accent: "#2ecc71",
  lightGray: "#f0f0f0",
  border: "#e1e4e8",
  amber: "#d97706",
};

const FONT_FAMILY =
  'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

const GICS_SECTOR_MAPPING = {
  "basic-materials": "Materials",
  "communication-services": "Communication Services",
  "consumer-cyclical": "Consumer Discretionary",
  "consumer-defensive": "Consumer Staples",
  energy: "Energy",
  "financial-services": "Financials",
  healthcare: "Health Care",
  industrials: "Industrials",
  "real-estate": "Real Estate",
  technology: "Information Technology",
  utilities: "Utilities",
};

const PLOTLY_CONFIG = { responsive: true, displaylogo: false };

const csvCache = new Map();

async function loadCsv(path) {
  if (csvCache.has(path)) return csvCache.get(path);
  const promise = fetch(path, { cache: "force-cache" })
    .then((r) => {
      if (!r.ok) throw new Error(`Failed to load ${path}: HTTP ${r.status}`);
      return r.text();
    })
    .then(
      (text) =>
        new Promise((resolve, reject) => {
          Papa.parse(text, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (result) => {
              if (result.errors && result.errors.length) {
                console.warn(`CSV parse warnings for ${path}:`, result.errors);
              }
              resolve(result.data);
            },
            error: reject,
          });
        }),
    );
  csvCache.set(path, promise);
  return promise;
}

function linearFit(xs, ys) {
  const points = [];
  for (let i = 0; i < xs.length; i++) {
    const x = Number(xs[i]);
    const y = Number(ys[i]);
    if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
  }
  if (points.length < 2) return null;
  const { m: slope, b: intercept } = ss.linearRegression(points);
  const predict = (x) => slope * x + intercept;
  let rSquared;
  try {
    rSquared = ss.rSquared(points, predict);
  } catch (_) {
    rSquared = NaN;
  }
  return { slope, intercept, rSquared, predict, n: points.length };
}

function r2Annotation(rSquared) {
  if (!Number.isFinite(rSquared)) {
    return {
      text: "R\u00b2 = N/A (insufficient variance)",
      color: COLORS.secondary,
      subtitle: "Low R\u00b2 indicates the fit is unreliable for this sector.",
    };
  }
  if (rSquared < 0.1) {
    return {
      text: `R\u00b2 = ${rSquared.toFixed(3)}`,
      color: COLORS.secondary,
      subtitle: "Low R\u00b2 indicates the fit is unreliable for this sector.",
    };
  }
  if (rSquared < 0.3) {
    return { text: `R\u00b2 = ${rSquared.toFixed(3)}`, color: COLORS.amber, subtitle: null };
  }
  return { text: `R\u00b2 = ${rSquared.toFixed(3)}`, color: COLORS.text, subtitle: null };
}

function htmlEscape(value) {
  if (value == null) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function scatterLayout({ title, height = 600 }) {
  return {
    title: {
      text: title,
      font: { family: FONT_FAMILY, size: 24, color: COLORS.text },
      x: 0.5,
      xanchor: "center",
    },
    xaxis: {
      title: { text: "Fundamental Z-score", font: { family: FONT_FAMILY } },
      showgrid: true,
      gridcolor: COLORS.lightGray,
      gridwidth: 1,
      zeroline: false,
      showline: true,
      linewidth: 1,
      linecolor: COLORS.border,
      mirror: true,
      tickfont: { family: FONT_FAMILY },
    },
    yaxis: {
      title: { text: "P/E Ratio", font: { family: FONT_FAMILY } },
      showgrid: true,
      gridcolor: COLORS.lightGray,
      gridwidth: 1,
      zeroline: false,
      showline: true,
      linewidth: 1,
      linecolor: COLORS.border,
      mirror: true,
      tickfont: { family: FONT_FAMILY },
    },
    legend: {
      font: { family: FONT_FAMILY },
      bgcolor: "rgba(255,255,255,0.8)",
      bordercolor: COLORS.border,
      borderwidth: 1,
    },
    showlegend: true,
    hovermode: "closest",
    font: { family: FONT_FAMILY },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    height,
    margin: { l: 40, r: 40, t: 80, b: 40 },
  };
}

function fitAnnotation(slope, intercept, rSquared) {
  const equation = `y = ${slope.toFixed(2)}x + ${intercept.toFixed(2)}`;
  const r2 = r2Annotation(rSquared);
  const lines = [equation, `<span style="color:${r2.color}">${r2.text}</span>`];
  if (r2.subtitle) {
    lines.push(`<span style="color:${r2.color};font-size:10px">${r2.subtitle}</span>`);
  }
  return {
    x: 0.02,
    y: 0.98,
    xref: "paper",
    yref: "paper",
    text: lines.join("<br>"),
    showarrow: false,
    font: { family: FONT_FAMILY, size: 12 },
    bgcolor: "rgba(255,255,255,0.8)",
    bordercolor: COLORS.border,
    borderwidth: 1,
    align: "left",
  };
}

function deviationBarLayout({ title, predicted, height, titleSize, sectorRows }) {
  const sectorPEs = sectorRows.map((r) => Number(r.PE)).filter(Number.isFinite);
  const sectorMax = Math.max(...sectorPEs);
  const sectorMin = Math.min(...sectorPEs);
  const range = sectorMax - sectorMin;
  const xMin = Math.max(0, sectorMin - range * 0.1);
  const xMax = sectorMax + range * 0.1;
  return {
    title: {
      text: title,
      font: { family: FONT_FAMILY, size: titleSize, color: COLORS.text },
      x: 0.5,
      xanchor: "center",
    },
    xaxis: {
      title: "P/E Ratio",
      range: [xMin, xMax],
      tickfont: { family: FONT_FAMILY, size: 12 },
      gridcolor: COLORS.lightGray,
    },
    yaxis: { showticklabels: false, fixedrange: true },
    height,
    margin: { l: 20, r: 20, t: 50, b: 30 },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    showlegend: false,
    shapes: [
      {
        type: "line",
        x0: predicted,
        x1: predicted,
        y0: -0.5,
        y1: 0.5,
        yref: "y",
        xref: "x",
        line: { color: COLORS.text, width: 2 },
      },
    ],
    annotations: [
      {
        x: predicted,
        y: 0.5,
        yref: "y",
        xref: "x",
        text: `Predicted: ${predicted.toFixed(1)}`,
        showarrow: false,
        yshift: 14,
        font: { family: FONT_FAMILY, size: 12, color: COLORS.text },
      },
    ],
  };
}

function deviationBarTraces({ actual, predicted }) {
  const peMin = Math.min(actual, predicted);
  const peMax = Math.max(actual, predicted);
  const deviation = actual - predicted;
  return [
    {
      x: [peMax - peMin],
      y: ["P/E Range"],
      type: "bar",
      orientation: "h",
      marker: { color: actual < predicted ? COLORS.accent : COLORS.secondary },
      base: [peMin],
      text: [`Actual: ${actual.toFixed(1)}`],
      textposition: "outside",
      hoverinfo: "text",
      hovertext: [
        `Actual P/E: ${actual.toFixed(2)}<br>Predicted P/E: ${predicted.toFixed(
          2,
        )}<br>Deviation: ${deviation.toFixed(2)}`,
      ],
    },
  ];
}

const sectorTab = {
  rows: null,
  bySector: new Map(),

  async init() {
    if (this.rows) return;
    const data = await loadCsv("/sector_analysis.csv");
    this.rows = data.filter((row) => row && row.Sector && Number.isFinite(Number(row.PE)));
    for (const row of this.rows) {
      if (!this.bySector.has(row.Sector)) this.bySector.set(row.Sector, []);
      this.bySector.get(row.Sector).push(row);
    }
    this.bindEvents();
    this.populateSectorSelect();
  },

  bindEvents() {
    document.getElementById("sector-select").addEventListener("change", () => {
      this.populateCompanySelect();
      this.render();
    });
    document.getElementById("company-select").addEventListener("change", () => this.render());
  },

  populateSectorSelect() {
    const select = document.getElementById("sector-select");
    const slugs = [...this.bySector.keys()].sort((a, b) =>
      (GICS_SECTOR_MAPPING[a] || a).localeCompare(GICS_SECTOR_MAPPING[b] || b),
    );
    select.innerHTML = slugs
      .map(
        (slug) =>
          `<option value="${htmlEscape(slug)}">${htmlEscape(GICS_SECTOR_MAPPING[slug] || slug)}</option>`,
      )
      .join("");
    if (slugs.length) select.value = slugs[0];
    this.populateCompanySelect();
    this.render();
  },

  populateCompanySelect() {
    const sectorSlug = document.getElementById("sector-select").value;
    const companies = (this.bySector.get(sectorSlug) || []).map((r) => r.Ticker);
    const select = document.getElementById("company-select");
    select.innerHTML = companies
      .map((t) => `<option value="${htmlEscape(t)}">${htmlEscape(t)}</option>`)
      .join("");
    if (companies.length) select.value = companies[0];
  },

  render() {
    const sectorSlug = document.getElementById("sector-select").value;
    const ticker = document.getElementById("company-select").value;
    const sectorRows = this.bySector.get(sectorSlug) || [];
    this.renderScatter(sectorSlug, sectorRows, ticker);
    this.renderCompanyInfo(sectorRows, ticker);
  },

  renderScatter(sectorSlug, sectorRows, selectedTicker) {
    const target = document.getElementById("sector-scatter");
    const sectorLabel = GICS_SECTOR_MAPPING[sectorSlug] || sectorSlug;

    if (!sectorRows.length) {
      Plotly.purge(target);
      Plotly.newPlot(
        target,
        [],
        { ...scatterLayout({ title: `No data available for ${sectorLabel}` }), showlegend: false },
        PLOTLY_CONFIG,
      );
      return;
    }

    const xs = sectorRows.map((r) => r.composite_z_score);
    const ys = sectorRows.map((r) => r.PE);
    const fit = linearFit(xs, ys);

    const traces = [
      {
        x: xs,
        y: ys,
        mode: "markers",
        type: "scatter",
        name: "Stocks",
        text: sectorRows.map(
          (r) =>
            `Ticker: ${r.Ticker}<br>P/E: ${Number(r.PE).toFixed(2)}<br>Fundamental Z-score: ${Number(
              r.composite_z_score,
            ).toFixed(2)}`,
        ),
        hoverinfo: "text",
        marker: {
          size: 10,
          color: COLORS.primary,
          line: { width: 1.5, color: "white" },
          opacity: 0.8,
        },
      },
    ];

    const annotations = [];
    if (fit) {
      const finiteXs = xs.map(Number).filter(Number.isFinite);
      const xMin = Math.min(...finiteXs);
      const xMax = Math.max(...finiteXs);
      traces.push({
        x: [xMin, xMax],
        y: [fit.predict(xMin), fit.predict(xMax)],
        mode: "lines",
        type: "scatter",
        name: "Line of Best Fit",
        line: { color: COLORS.secondary, dash: "dash", width: 2 },
        hoverinfo: "skip",
      });
      annotations.push(fitAnnotation(fit.slope, fit.intercept, fit.rSquared));
    }

    const selected = sectorRows.find((r) => r.Ticker === selectedTicker);
    if (selected && Number.isFinite(Number(selected.composite_z_score)) && Number.isFinite(Number(selected.PE))) {
      traces.push({
        x: [Number(selected.composite_z_score)],
        y: [Number(selected.PE)],
        mode: "markers",
        type: "scatter",
        name: "Selected Company",
        marker: {
          size: 14,
          color: COLORS.accent,
          line: { width: 2, color: "white" },
          symbol: "star",
        },
        hoverinfo: "skip",
      });
    }

    const layout = {
      ...scatterLayout({ title: `Fundamental Z-score vs P/E Ratio - ${sectorLabel}` }),
      annotations,
    };

    Plotly.newPlot(target, traces, layout, PLOTLY_CONFIG);
  },

  renderCompanyInfo(sectorRows, selectedTicker) {
    const target = document.getElementById("sector-company-info");
    const row = sectorRows.find((r) => r.Ticker === selectedTicker);
    if (!row) {
      target.innerHTML = "";
      return;
    }

    const xs = sectorRows.map((r) => r.composite_z_score);
    const ys = sectorRows.map((r) => r.PE);
    const fit = linearFit(xs, ys);
    if (!fit) {
      target.innerHTML = "";
      return;
    }
    const z = Number(row.composite_z_score);
    const actual = Number(row.PE);
    const predicted = fit.predict(z);

    target.innerHTML = `
      <div class="info-block">
        <h3>${htmlEscape(row.Ticker)}</h3>
        <p>P/E Ratio: ${actual.toFixed(2)}</p>
        <p>Predicted P/E: ${predicted.toFixed(2)}</p>
        <p>Fundamental Z-score: ${z.toFixed(2)}</p>
        <div id="sector-deviation" class="plot plot-short" style="min-height:150px"></div>
      </div>
    `;

    Plotly.newPlot(
      document.getElementById("sector-deviation"),
      deviationBarTraces({ actual, predicted }),
      deviationBarLayout({
        title: "Actual vs. Predicted P/E Ratio",
        predicted,
        height: 150,
        titleSize: 16,
        sectorRows,
      }),
      PLOTLY_CONFIG,
    );
  },
};

function renderBacktestFooter() {
  const el = document.getElementById("backtest-footer");
  if (!el) return;
  el.innerHTML = "";
}

document.addEventListener("DOMContentLoaded", () => {
  for (const radio of document.querySelectorAll(".tab-radio")) {
    radio.addEventListener("change", () => {});
  }
  sectorTab.init().catch((err) => {
    console.error("Failed to initialise Sector Analysis tab", err);
    const target = document.getElementById("sector-scatter");
    target.innerHTML = `<p style="color:${COLORS.secondary};padding:24px">Failed to load sector_analysis.csv. ${htmlEscape(err.message)}</p>`;
  });
  renderBacktestFooter();
});
