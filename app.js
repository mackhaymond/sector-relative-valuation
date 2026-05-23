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

// ---------------------------------------------------------------------
// Linear algebra + distribution helpers for the Factor Selection tab.
// Multivariate OLS isn't in simple-statistics; the routines below solve
// (X^T X) beta = X^T y by Gaussian elimination with partial pivoting,
// and back out two-sided t- and F-test p-values via the regularized
// incomplete beta function (Numerical Recipes 6.4 betacf + lgamma).
// Accurate to ~6 significant figures - enough for any p-value a
// reader of a 'statsmodels-style' summary would care about.
// ---------------------------------------------------------------------

function matrixTranspose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const T = Array.from({ length: cols }, () => new Array(rows));
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) T[j][i] = A[i][j];
  return T;
}

function matMul(A, B) {
  const m = A.length;
  const n = B[0].length;
  const k = B.length;
  const C = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let r = 0; r < k; r++) s += A[i][r] * B[r][j];
      C[i][j] = s;
    }
  }
  return C;
}

function matVec(A, v) {
  const m = A.length;
  const out = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < v.length; j++) s += A[i][j] * v[j];
    out[i] = s;
  }
  return out;
}

function invertSymmetric(A) {
  const n = A.length;
  const aug = Array.from({ length: n }, (_, i) => {
    const row = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n + i] = 1;
    return row;
  });
  for (let i = 0; i < n; i++) {
    let pivotRow = i;
    let pivotMag = Math.abs(aug[i][i]);
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(aug[r][i]) > pivotMag) {
        pivotMag = Math.abs(aug[r][i]);
        pivotRow = r;
      }
    }
    if (pivotMag < 1e-12) throw new Error("Singular matrix - factor collinearity is severe");
    if (pivotRow !== i) [aug[i], aug[pivotRow]] = [aug[pivotRow], aug[i]];
    const pivot = aug[i][i];
    for (let j = 0; j < 2 * n; j++) aug[i][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = aug[r][i];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) aug[r][j] -= factor * aug[i][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

function lgamma(x) {
  const c = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 1.208650973866179e-3, -5.395239384953e-6,
  ];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) {
    y += 1;
    ser += c[j] / y;
  }
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

function betacf(a, b, x) {
  const FPMIN = 1e-300;
  const EPS = 3e-7;
  const maxIters = 200;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= maxIters; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) return h;
  }
  return h;
}

function regularizedIncompleteBeta(a, b, x) {
  if (x < 0 || x > 1) return Number.NaN;
  if (x === 0 || x === 1) return x;
  const bt = Math.exp(
    lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x),
  );
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(a, b, x)) / a;
  }
  return 1 - (bt * betacf(b, a, 1 - x)) / b;
}

function studentTTwoSidedPValue(t, df) {
  if (!Number.isFinite(t) || df <= 0) return Number.NaN;
  const x = df / (df + t * t);
  return regularizedIncompleteBeta(df / 2, 0.5, x);
}

function fDistributionPValue(f, d1, d2) {
  if (!Number.isFinite(f) || f < 0 || d1 <= 0 || d2 <= 0) return Number.NaN;
  const x = d2 / (d2 + d1 * f);
  return regularizedIncompleteBeta(d2 / 2, d1 / 2, x);
}

function fitOls(X, y) {
  const n = X.length;
  const p = X[0].length;
  if (n <= p) throw new Error(`Need n > p (n=${n}, p=${p})`);
  const Xt = matrixTranspose(X);
  const XtX = matMul(Xt, X);
  const XtY = matVec(Xt, y);
  const XtXInv = invertSymmetric(XtX);
  const beta = matVec(XtXInv, XtY);
  const yHat = matVec(X, beta);
  const residuals = y.map((yi, i) => yi - yHat[i]);
  const ssRes = residuals.reduce((acc, e) => acc + e * e, 0);
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const ssTot = y.reduce((acc, yi) => acc + (yi - yMean) ** 2, 0);
  const rSquared = ssTot === 0 ? Number.NaN : 1 - ssRes / ssTot;
  const dfModel = p - 1;
  const dfResid = n - p;
  const sigma2 = ssRes / dfResid;
  const adjR2 = Number.isFinite(rSquared)
    ? 1 - (1 - rSquared) * ((n - 1) / dfResid)
    : Number.NaN;
  const fStat =
    dfModel > 0 && ssTot > 0 ? ((ssTot - ssRes) / dfModel) / sigma2 : Number.NaN;
  const fPValue = Number.isFinite(fStat) ? fDistributionPValue(fStat, dfModel, dfResid) : Number.NaN;
  const standardErrors = XtXInv.map((row, i) => Math.sqrt(sigma2 * row[i]));
  const tStats = beta.map((b, i) => (standardErrors[i] === 0 ? Number.NaN : b / standardErrors[i]));
  const pValues = tStats.map((t) => studentTTwoSidedPValue(t, dfResid));
  return {
    n,
    p,
    dfModel,
    dfResid,
    beta,
    standardErrors,
    tStats,
    pValues,
    rSquared,
    adjR2,
    fStat,
    fPValue,
    sigma: Math.sqrt(sigma2),
  };
}

const FACTOR_GROUPS = ["risk", "momentum", "quality", "size", "growth"];

const GROUP_TO_COLUMN = {
  risk: "Risk_Score",
  momentum: "Momentum_Score",
  quality: "Quality_Score",
  size: "Size_Score",
  growth: "Growth_Score",
};

const GROUP_TO_METRICS = {
  risk: ["MaxDrawdown", "DebtToEquity", "ReturnSD"],
  momentum: ["PriceChange12M", "RSI", "EarningsGrowth"],
  quality: ["ROE", "ROA", "OperatingMargin", "EBITDAMargin"],
  size: ["LogMarketCap"],
  growth: ["RevenueGrowth"],
};

const factorTab = {
  initialized: false,

  async init() {
    if (this.initialized) return;
    this.initialized = true;
    await sectorTab.init();
    this.bindMetricToggles();
    document.getElementById("recalculate-button").addEventListener("click", () => this.run());
    this.run();
  },

  bindMetricToggles() {
    for (const group of FACTOR_GROUPS) {
      const cb = document.querySelector(`.factor-checkbox[value="${group}"]`);
      if (!cb) continue;
      cb.addEventListener("change", () => {
        const display = document.querySelector(`.metrics-display[data-group="${group}"]`);
        if (display) display.classList.toggle("is-hidden", !cb.checked);
      });
    }
  },

  selectedGroups() {
    const selected = FACTOR_GROUPS.filter((g) => {
      const cb = document.querySelector(`.factor-checkbox[value="${g}"]`);
      return cb && cb.checked;
    });
    return selected.length ? selected : [...FACTOR_GROUPS];
  },

  run() {
    const output = document.getElementById("regression-output");
    const groups = this.selectedGroups();
    const cols = groups.map((g) => GROUP_TO_COLUMN[g]);

    const rows = (sectorTab.rows || []).filter(
      (r) => Number.isFinite(Number(r.PE)) && cols.every((c) => Number.isFinite(Number(r[c]))),
    );
    if (rows.length === 0) {
      output.textContent = "No rows available for the selected factor groups.";
      return;
    }

    const X = rows.map((r) => [1, ...cols.map((c) => Number(r[c]))]);
    const y = rows.map((r) => Number(r.PE));

    let result;
    try {
      result = fitOls(X, y);
    } catch (err) {
      output.textContent = `Regression failed: ${err.message}`;
      return;
    }

    output.textContent = this.formatSummary(groups, cols, result);
  },

  formatSummary(groups, cols, r) {
    const lines = [
      "OLS Regression Results for PE Ratio prediction",
      "=====================================",
      "",
      "Selected Factor Groups:",
    ];
    for (const g of groups) {
      lines.push(`- ${capitalize(g)}: ${GROUP_TO_METRICS[g].join(", ")}`);
    }
    lines.push("");
    lines.push("Regression Statistics:");
    lines.push(`R-squared: ${formatNumber(r.rSquared, 4)}`);
    lines.push(`Adjusted R-squared: ${formatNumber(r.adjR2, 4)}`);
    lines.push(`F-statistic: ${formatNumber(r.fStat, 4)}`);
    lines.push(`Prob (F-statistic): ${formatNumber(r.fPValue, 4)}`);
    lines.push("");
    lines.push("Coefficients:");
    const paramNames = ["const", ...cols];
    for (let i = 0; i < paramNames.length; i++) {
      lines.push(
        `${paramNames[i]}: ${formatNumber(r.beta[i], 4)} (p=${formatNumber(r.pValues[i], 4)})`,
      );
    }
    lines.push("");
    lines.push("Full Statistical Summary:");
    lines.push("----------------------------");
    lines.push(this.formatStatsmodelsTable(paramNames, r));
    return lines.join("\n");
  },

  formatStatsmodelsTable(paramNames, r) {
    const header = [
      `${"".padEnd(20)} ${"coef".padStart(12)} ${"std err".padStart(12)} ${"t".padStart(10)} ${"P>|t|".padStart(10)}`,
      `${"-".repeat(70)}`,
    ];
    const body = paramNames.map(
      (name, i) =>
        `${name.padEnd(20)} ${formatNumber(r.beta[i], 4).padStart(12)} ${formatNumber(r.standardErrors[i], 4).padStart(12)} ${formatNumber(r.tStats[i], 3).padStart(10)} ${formatNumber(r.pValues[i], 3).padStart(10)}`,
    );
    const footer = [
      `${"-".repeat(70)}`,
      `Observations: ${r.n}`,
      `Df Residuals: ${r.dfResid}`,
      `Df Model:     ${r.dfModel}`,
      `Sigma (RMSE): ${formatNumber(r.sigma, 4)}`,
    ];
    return [...header, ...body, ...footer].join("\n");
  },
};

function capitalize(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
}

function formatNumber(n, digits) {
  if (!Number.isFinite(n)) return "NaN";
  if (Math.abs(n) >= 1e6 || (Math.abs(n) < 1e-4 && n !== 0)) return n.toExponential(digits);
  return n.toFixed(digits);
}

const REPO_URL = "https://github.com/mackhaymond/sector-relative-valuation";

function renderBacktestFooter() {
  const el = document.getElementById("backtest-footer");
  if (!el) return;
  el.innerHTML = `
    <h4>36-month PIT backtest</h4>
    <p>
      May 2023 \u2013 May 2026, Russell 1000, monthly rebalance, 10 bps round-trip cost:
      mean IC <strong>\u22120.013</strong> (t = \u22121.17),
      long-short Sharpe <strong>0.05</strong>,
      cumulative return <strong>+0.43%</strong>.
    </p>
    <p>The deviation signal does <strong>not</strong> predict 1-month forward returns at this universe and window.</p>
    <p class="footer-links">
      Methodology and limitations:
      <a href="${REPO_URL}/blob/main/BACKTEST.md" target="_blank" rel="noopener">BACKTEST.md</a>
      &middot;
      <a href="${REPO_URL}/blob/main/backtest_artifacts/cumulative_long_short_return.png" target="_blank" rel="noopener">Cumulative-return chart</a>
      &middot;
      <a href="${REPO_URL}" target="_blank" rel="noopener">Source</a>
    </p>
  `;
}

document.addEventListener("DOMContentLoaded", () => {
  sectorTab.init().catch((err) => {
    console.error("Failed to initialise Sector Analysis tab", err);
    const target = document.getElementById("sector-scatter");
    target.innerHTML = `<p style="color:${COLORS.secondary};padding:24px">Failed to load sector_analysis.csv. ${htmlEscape(err.message)}</p>`;
  });
  factorTab.init().catch((err) => {
    console.error("Failed to initialise Factor Selection tab", err);
    const target = document.getElementById("regression-output");
    if (target) target.textContent = `Failed to initialise: ${err.message}`;
  });
  renderBacktestFooter();
});
