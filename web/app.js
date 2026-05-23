/*
 * Sector-Relative Valuation - browser app.
 *
 * Ports the Dash dashboard in src/dashboard.py to vanilla JS + Plotly.
 * Loads three CSVs from the Pages asset root:
 *   - /sector_analysis.csv         (per-ticker composites + PE + composite_z_score)
 *   - /sector_analysis_full.csv    (per-ticker raw metrics + every z-score)
 *   - /weights.csv                 (per-sector Ridge weights + alpha + r_squared)
 *
 * Lazy-loads on tab change: the Individual Stock and Factor tabs only
 * fetch the CSVs they need the first time they're activated. Phase 1
 * is scaffold-only; later phases wire the interactions.
 */

"use strict";

// ---------------------------------------------------------------------
// Constants and color palette - kept in lockstep with web/styles.css so
// that Plotly traces match the surrounding chrome.
// ---------------------------------------------------------------------

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

// Display labels for sector slugs. Mirrors GICS_SECTOR_MAPPING in
// src/dashboard.py so users see the same names across the two
// implementations.
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

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  // Wire tabs - radio inputs handle the visual switch via CSS; this
  // listener exists to trigger lazy-loading once a tab is first opened.
  // Concrete handlers attach in later phases.
  for (const radio of document.querySelectorAll(".tab-radio")) {
    radio.addEventListener("change", onTabChange);
  }

  renderBacktestFooter();
});

function onTabChange(event) {
  // Phase 1 stub - lazy-load and render are wired in phases 2-4.
  // Kept as a noop here so the radio focus + change cycle works
  // end-to-end before the data plumbing exists.
  void event;
}

// ---------------------------------------------------------------------
// Backtest footer
// ---------------------------------------------------------------------

function renderBacktestFooter() {
  // Phase 5 hardcodes the headline numbers; placeholder lives here so
  // Phase 1 has the markup hook visible. Real content lands in Phase 5.
  const el = document.getElementById("backtest-footer");
  if (!el) return;
  el.innerHTML = "";
}
