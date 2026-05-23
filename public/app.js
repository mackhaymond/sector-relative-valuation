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

/*
 * Tab 3: Individual Stock Analysis.
 *
 * Fetches a single ticker's live fundamentals via /api/yf, computes
 * per-category z-scores against the sector cross-section in
 * sector_analysis_full.csv, builds the composite z-score against the
 * sector weights in weights.csv (with category_is_missing renormalization),
 * fits a sector-local OLS line of PE vs composite, and renders the
 * scatter + deviation bar + info card. Port of analyze_individual_stock
 * at src/dashboard.py:880.
 *
 * Honesty pattern: categories whose underlying metrics ALL came back
 * unresolved are flagged as 'unavailable' in the info card rather than
 * substituted with a 0.0 z-score that would silently read as
 * 'exactly average'. Same for actual P/E when Yahoo returned null.
 */

const CATEGORY_TO_METRICS = {
  Risk_Score: ["MaxDrawdown", "DebtToEquity", "ReturnSD"],
  Momentum_Score: ["PriceChange12M", "RSI", "EarningsGrowth"],
  Quality_Score: ["ROE", "ROA", "OperatingMargin", "EBITDAMargin"],
  Size_Score: ["LogMarketCap"],
  Growth_Score: ["RevenueGrowth"],
};

const CATEGORY_LABEL = {
  Risk_Score: "Risk Score",
  Momentum_Score: "Momentum Score",
  Quality_Score: "Quality Score",
  Size_Score: "Size Score",
  Growth_Score: "Growth Score",
};

function sectorSlugFromName(name) {
  return (name || "").toString().toLowerCase().replace(/\s+/g, "-");
}

// 30-period SIMPLE rolling-mean RSI - mirrors src/data.py:calculate_rsi.
// Wilder's exponential smoothing would produce a different value, which
// would mis-z-score against the sector cross-section that was built
// with the simple rolling-mean variant.
function calculateRSI(closes, periods = 30) {
  if (!Array.isArray(closes) || closes.length <= periods) return NaN;
  const gains = new Array(closes.length - 1);
  const losses = new Array(closes.length - 1);
  for (let i = 1; i < closes.length; i++) {
    const delta = closes[i] - closes[i - 1];
    gains[i - 1] = delta > 0 ? delta : 0;
    losses[i - 1] = delta < 0 ? -delta : 0;
  }
  if (gains.length < periods) return NaN;
  let gainSum = 0;
  let lossSum = 0;
  for (let i = gains.length - periods; i < gains.length; i++) {
    gainSum += gains[i];
    lossSum += losses[i];
  }
  const avgGain = gainSum / periods;
  const avgLoss = lossSum / periods;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

function calculateReturnSD(closes) {
  if (!Array.isArray(closes) || closes.length < 2) return NaN;
  const returns = [];
  for (let i = 1; i < closes.length; i++) {
    if (closes[i - 1] === 0 || !Number.isFinite(closes[i - 1])) continue;
    returns.push(closes[i] / closes[i - 1] - 1);
  }
  if (returns.length < 2) return NaN;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance =
    returns.reduce((acc, r) => acc + (r - mean) ** 2, 0) / (returns.length - 1);
  return Math.sqrt(variance);
}

function calculateMaxDrawdown(closes) {
  if (!Array.isArray(closes) || closes.length < 2) return NaN;
  let peak = closes[0];
  let maxDD = 0;
  for (const px of closes) {
    if (!Number.isFinite(px)) continue;
    if (px > peak) peak = px;
    if (peak > 0) {
      const dd = (px - peak) / peak;
      if (dd < maxDD) maxDD = dd;
    }
  }
  return Math.abs(maxDD);
}

function extractMetricsFromUpstream(payload) {
  const summary = payload.summary || {};
  const quote = payload.quote || {};
  const history = Array.isArray(payload.history) ? payload.history : [];

  const fin = summary.financialData || {};
  const dks = summary.defaultKeyStatistics || {};
  const price = summary.price || {};
  const detail = summary.summaryDetail || {};

  const closes = history
    .map((h) => Number(h && h.close))
    .filter(Number.isFinite);

  const marketCap = Number(
    price.marketCap ?? quote.marketCap ?? detail.marketCap ?? NaN,
  );
  const logMarketCap =
    Number.isFinite(marketCap) && marketCap > 0 ? Math.log(marketCap) : NaN;

  const peCandidates = [
    quote.trailingPE,
    detail.trailingPE,
    dks.trailingEps && quote.regularMarketPrice
      ? quote.regularMarketPrice / dks.trailingEps
      : undefined,
    quote.forwardPE,
    dks.forwardPE,
  ];
  const actualPE = Number(peCandidates.find((v) => Number.isFinite(Number(v))));

  return {
    metrics: {
      MaxDrawdown: closes.length ? calculateMaxDrawdown(closes) : NaN,
      DebtToEquity: Number(fin.debtToEquity ?? NaN),
      ReturnSD: closes.length ? calculateReturnSD(closes) : NaN,
      PriceChange12M: Number(
        dks["52WeekChange"] ?? quote.fiftyTwoWeekChangePercent ?? NaN,
      ),
      RSI: closes.length ? calculateRSI(closes) : NaN,
      EarningsGrowth: Number(fin.earningsGrowth ?? NaN),
      ROE: Number(fin.returnOnEquity ?? NaN),
      ROA: Number(fin.returnOnAssets ?? NaN),
      OperatingMargin: Number(fin.operatingMargins ?? NaN),
      EBITDAMargin: Number(
        fin.ebitdaMargins ?? dks.ebitdaMargins ?? detail.ebitdaMargins ?? NaN,
      ),
      LogMarketCap: logMarketCap,
      RevenueGrowth: Number(fin.revenueGrowth ?? NaN),
    },
    actualPE: Number.isFinite(actualPE) ? actualPE : NaN,
    sectorRaw: (summary.assetProfile && summary.assetProfile.sector) || quote.sector || "",
    longName: price.longName || price.shortName || quote.longName || quote.shortName || payload.ticker,
  };
}

// Mirrors src/dashboard.py:_zscore_against_sector. Uses sample std
// (ddof=1) - matches pandas Series.std() default rather than scipy's
// zscore default of ddof=0. The Python dashboard makes the same choice.
function zScoreAgainstSector(rows, metricName, value) {
  if (!Number.isFinite(value) || !Array.isArray(rows) || rows.length === 0) return NaN;
  const values = rows.map((r) => Number(r[metricName])).filter(Number.isFinite);
  if (values.length < 2) return NaN;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (values.length - 1);
  const std = Math.sqrt(variance);
  if (std === 0) return NaN;
  return (value - mean) / std;
}

const individualTab = {
  initialized: false,
  fullRows: null,
  weightsBySector: null,

  async init() {
    if (this.initialized) return;
    this.initialized = true;
    const [fullRows, weightsRows] = await Promise.all([
      loadCsv("/sector_analysis_full.csv"),
      loadCsv("/weights.csv"),
    ]);
    this.fullRows = fullRows.filter((r) => r && r.Sector);
    this.weightsBySector = new Map(
      weightsRows.filter((r) => r && r.Sector).map((r) => [r.Sector, r]),
    );
    this.bindEvents();
  },

  bindEvents() {
    const button = document.getElementById("analyze-button");
    const input = document.getElementById("ticker-input");
    if (!button || !input) return;
    button.addEventListener("click", () => this.analyze());
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        this.analyze();
      }
    });
  },

  setStatus(html, isError = false) {
    const el = document.getElementById("analysis-status");
    if (!el) return;
    el.innerHTML = html;
    el.classList.toggle("status-error", isError);
  },

  setError(ticker, message) {
    this.setStatus(
      `Error analyzing <span class="status-success-ticker">${htmlEscape(ticker)}</span>: ${htmlEscape(message)}`,
      true,
    );
    Plotly.purge(document.getElementById("individual-scatter"));
    Plotly.purge(document.getElementById("individual-deviation"));
    document.getElementById("individual-info").innerHTML = "";
  },

  async analyze() {
    const input = document.getElementById("ticker-input");
    const ticker = (input.value || "").trim().toUpperCase();
    if (!ticker) {
      this.setStatus("Enter a ticker symbol above and click Analyze.");
      return;
    }
    this.setStatus(`Fetching <span class="status-success-ticker">${htmlEscape(ticker)}</span>...`);

    let payload;
    try {
      const res = await fetch(`/api/yf?ticker=${encodeURIComponent(ticker)}`);
      payload = await res.json();
      if (!res.ok) {
        this.setError(ticker, payload.error || `Upstream error (HTTP ${res.status})`);
        return;
      }
    } catch (err) {
      this.setError(ticker, err.message || "Network error");
      return;
    }

    const { metrics, actualPE, sectorRaw, longName } = extractMetricsFromUpstream(payload);
    const sectorSlug = sectorSlugFromName(sectorRaw);
    if (!sectorSlug || !this.weightsBySector.has(sectorSlug)) {
      this.setError(ticker, `Could not determine sector for ${ticker} (got "${sectorRaw}")`);
      return;
    }

    const sectorRows = this.fullRows.filter((r) => r.Sector === sectorSlug);
    if (sectorRows.length === 0) {
      this.setError(ticker, `No reference data for sector "${sectorSlug}"`);
      return;
    }
    const sectorWeights = this.weightsBySector.get(sectorSlug);

    const categoryScores = {};
    const categoryIsMissing = {};
    for (const [category, metricNames] of Object.entries(CATEGORY_TO_METRICS)) {
      const zs = [];
      for (const m of metricNames) {
        const z = zScoreAgainstSector(sectorRows, m, metrics[m]);
        if (Number.isFinite(z)) zs.push(z);
      }
      if (zs.length > 0) {
        categoryScores[category] = zs.reduce((a, b) => a + b, 0) / zs.length;
        categoryIsMissing[category] = false;
      } else {
        categoryScores[category] = 0;
        categoryIsMissing[category] = true;
      }
    }

    if (Object.values(categoryIsMissing).every((v) => v)) {
      this.setError(ticker, "no fundamental data resolved from Yahoo Finance");
      return;
    }

    let weightedSum = 0;
    let totalWeight = 0;
    for (const category of Object.keys(CATEGORY_TO_METRICS)) {
      if (categoryIsMissing[category]) continue;
      const w = Number(sectorWeights[category]) / 100;
      if (!Number.isFinite(w)) continue;
      weightedSum += categoryScores[category] * w;
      totalWeight += w;
    }
    const compositeZ =
      totalWeight > 0
        ? weightedSum / totalWeight
        : Object.values(categoryScores).reduce((a, b) => a + b, 0) /
          Object.keys(categoryScores).length;

    const sectorPlotRows = this.computeSectorPlotRows(sectorRows, sectorWeights);
    const xs = sectorPlotRows.map((r) => r.composite);
    const ys = sectorPlotRows.map((r) => r.pe);
    const fit = linearFit(xs, ys);
    const predictedPE = fit ? fit.predict(compositeZ) : NaN;
    const peIsMissing = !Number.isFinite(actualPE);
    const peForChart = peIsMissing ? predictedPE : actualPE;
    const deviation = peIsMissing ? NaN : actualPE - predictedPE;

    this.renderScatter({
      ticker,
      longName,
      sectorSlug,
      sectorRows: sectorPlotRows,
      fit,
      compositeZ,
      peForChart,
    });
    this.renderDeviation({
      sectorRows: sectorPlotRows,
      actual: peForChart,
      predicted: predictedPE,
    });
    this.renderInfoCard({
      ticker,
      sectorSlug,
      compositeZ,
      actualPE,
      predictedPE,
      deviation,
      peIsMissing,
      categoryScores,
      categoryIsMissing,
    });

    const summary = `Analysis complete for <span class="status-success-ticker">${htmlEscape(ticker)}</span> &middot; Sector: ${htmlEscape(GICS_SECTOR_MAPPING[sectorSlug] || sectorSlug)} &middot; Fundamental Z-score: ${compositeZ.toFixed(2)}`;
    this.setStatus(summary);
  },

  computeSectorPlotRows(sectorRows, sectorWeights) {
    const out = [];
    for (const row of sectorRows) {
      const pe = Number(row.PE);
      if (!Number.isFinite(pe)) continue;
      let weightedSum = 0;
      let totalWeight = 0;
      for (const category of Object.keys(CATEGORY_TO_METRICS)) {
        const score = Number(row[category]);
        if (!Number.isFinite(score)) continue;
        const w = Number(sectorWeights[category]) / 100;
        if (!Number.isFinite(w)) continue;
        weightedSum += score * w;
        totalWeight += w;
      }
      if (totalWeight === 0) continue;
      out.push({ ticker: row.Ticker, composite: weightedSum / totalWeight, pe });
    }
    return out;
  },

  renderScatter({ ticker, sectorSlug, sectorRows, fit, compositeZ, peForChart }) {
    const target = document.getElementById("individual-scatter");
    const sectorLabel = GICS_SECTOR_MAPPING[sectorSlug] || sectorSlug;
    const traces = [
      {
        x: sectorRows.map((r) => r.composite),
        y: sectorRows.map((r) => r.pe),
        mode: "markers",
        type: "scatter",
        name: "Sector Stocks",
        text: sectorRows.map((r) => r.ticker),
        hovertemplate: "%{text}<br>Fundamental Z-score: %{x:.2f}<br>P/E: %{y:.2f}<extra></extra>",
        marker: { size: 10, color: COLORS.primary, opacity: 0.5 },
      },
    ];
    const annotations = [];
    if (fit) {
      const finiteXs = sectorRows.map((r) => r.composite).filter(Number.isFinite);
      const xMin = Math.min(...finiteXs, compositeZ);
      const xMax = Math.max(...finiteXs, compositeZ);
      traces.push({
        x: [xMin, xMax],
        y: [fit.predict(xMin), fit.predict(xMax)],
        mode: "lines",
        type: "scatter",
        name: "Sector Trend",
        line: { color: COLORS.secondary, dash: "dash" },
        hoverinfo: "skip",
      });
      annotations.push(fitAnnotation(fit.slope, fit.intercept, fit.rSquared));
    }
    traces.push({
      x: [compositeZ],
      y: [peForChart],
      mode: "markers",
      type: "scatter",
      name: ticker,
      text: [ticker],
      hovertemplate: "%{text}<br>Fundamental Z-score: %{x:.2f}<br>P/E: %{y:.2f}<extra></extra>",
      marker: { size: 15, color: COLORS.accent, line: { width: 2, color: "white" } },
    });
    const layout = {
      ...scatterLayout({ title: `${ticker} vs ${sectorLabel} Sector` }),
      annotations,
    };
    Plotly.newPlot(target, traces, layout, PLOTLY_CONFIG);
  },

  renderDeviation({ sectorRows, actual, predicted }) {
    const target = document.getElementById("individual-deviation");
    if (!Number.isFinite(actual) || !Number.isFinite(predicted)) {
      Plotly.purge(target);
      return;
    }
    Plotly.newPlot(
      target,
      deviationBarTraces({ actual, predicted }),
      deviationBarLayout({
        title: "P/E Ratio Analysis",
        predicted,
        height: 200,
        titleSize: 20,
        sectorRows: sectorRows.map((r) => ({ PE: r.pe })),
      }),
      PLOTLY_CONFIG,
    );
  },

  renderInfoCard({
    ticker,
    sectorSlug,
    compositeZ,
    actualPE,
    predictedPE,
    deviation,
    peIsMissing,
    categoryScores,
    categoryIsMissing,
  }) {
    const target = document.getElementById("individual-info");
    const sectorLabel = GICS_SECTOR_MAPPING[sectorSlug] || sectorSlug;
    const sections = [
      [
        "Company Information",
        [
          ["Sector", htmlEscape(sectorLabel), false],
          ["Fundamental Z-score", compositeZ.toFixed(2), false],
        ],
      ],
      [
        "P/E Analysis",
        [
          ["Actual P/E", peIsMissing ? "unavailable" : actualPE.toFixed(2), peIsMissing],
          ["Predicted P/E", Number.isFinite(predictedPE) ? predictedPE.toFixed(2) : "unavailable", !Number.isFinite(predictedPE)],
          ["P/E Deviation", peIsMissing || !Number.isFinite(deviation) ? "unavailable" : deviation.toFixed(2), peIsMissing || !Number.isFinite(deviation)],
        ],
      ],
      [
        "Category Scores",
        Object.keys(CATEGORY_TO_METRICS).map((cat) => [
          CATEGORY_LABEL[cat],
          categoryIsMissing[cat] ? "unavailable" : categoryScores[cat].toFixed(2),
          categoryIsMissing[cat],
        ]),
      ],
    ];
    const header = `<h3>${htmlEscape(ticker)} Analysis</h3>`;
    const body = sections
      .map(
        ([sectionTitle, rows]) => `
          <h4>${htmlEscape(sectionTitle)}</h4>
          ${rows
            .map(
              ([label, value, isMissing]) => `
                <div class="info-row">
                  <span class="info-label">${htmlEscape(label)}:</span>
                  <span class="${isMissing ? "info-value-unavailable" : ""}">${value}</span>
                </div>`,
            )
            .join("")}
        `,
      )
      .join("");
    target.innerHTML = `<div class="info-block">${header}${body}</div>`;
  },
};

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
  individualTab.init().catch((err) => {
    console.error("Failed to initialise Individual Stock tab", err);
  });
  renderBacktestFooter();
});
