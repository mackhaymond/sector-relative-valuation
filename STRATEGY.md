# Strategy and Methodology

This document describes how the model is constructed and the reasoning behind each design choice. It is intentionally specific so that a reader who wants to challenge the methodology can locate the assumption they want to push back on.

## 1. Why sector-relative valuation

The headline problem this project addresses is that cross-sector comparisons of trailing P/E ratios are not meaningful. Sectors differ structurally in:

- **Capital intensity and depreciation schedules** — utilities and energy carry meaningfully different earnings volatility and reinvestment burdens than asset-light software.
- **Growth expectations baked into multiples** — communication services and technology typically trade on forward growth, real estate and consumer staples on yield and cash conversion.
- **Cost of capital** — financials' multiples are sensitive to rate regimes in a way that consumer cyclical multiples are not.

Ranking AAPL against XOM on P/E ranks Apple's sector against ExxonMobil's sector, not the two companies. To compare companies on valuation, the unit of comparison has to be *within* the sector — the company versus its own peers. Every step in the pipeline is built around that constraint.

## 2. Factor universe

Five factor categories are collected in `src/data.py` and all five are weighted in the production fit:

| Category   | Metrics                                                                           | Notes                                                              |
| ---------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Risk       | Max drawdown (1y), debt-to-equity, return standard deviation (1y)                 | Computed from 1y price history (max drawdown, return SD) and the yfinance `info` field for D/E. |
| Momentum   | 52-week price change, RSI, earnings growth                                        | RSI is the standard 30-period RSI on close prices.                  |
| Quality    | ROE, ROA, operating margin, EBITDA margin                                         | EBITDA margin uses the plural `ebitdaMargins` field; the singular form returns 100% null. Banks return `ebitdaMargins=0.0` systematically, which collapses to a NaN within-sector z-score and falls out of the composite mean. |
| Size       | log(market cap)                                                                   | Single derived metric: `ln(marketCap)`. Raw market cap, total assets and enterprise value were dropped (raw cap spans 6+ orders of magnitude; total assets is not on the info dict; EV is ~0.95+ correlated with market cap). |
| Growth     | Revenue growth                                                                    | Single-metric composite. yfinance does not populate `epsGrowth` or `cashFlowGrowth` on the info dict (both came back 100% null across the Russell 1000); only `revenueGrowth` survives the per-metric null-rate audit at ~99.6%. |

An earlier iteration of this codebase collected a separate Profitability category (gross / EBITDA / net profit margin) and two further categories — Value (price-to-book, EV/EBITDA, price-to-sales) and Liquidity (current ratio, quick ratio, interest coverage). Profitability was dropped because it double-counts Quality at a different income-statement level (EBITDA margin folded into Quality instead). Value was dropped because it is mechanically circular with the trailing-P/E regression target. Liquidity was dropped because it is a weak P/E predictor that breaks for financials. All three can be re-introduced if the design ever calls for them.

The target variable is the trailing P/E ratio. The framework supports alternative valuation targets (price-to-book, price-to-sales, price-to-free-cashflow) by changing the `Y_VALUATION_METRIC` map; only trailing P/E is exercised in the current fit.

## 3. Within-sector z-scoring

Every metric is standardized within its sector before being combined. For metric *m* in sector *s*:

```
z_{m,s,i} = (x_{m,s,i} - mean_{m,s}) / std_{m,s}
```

Missing values are mean-imputed within the sector before the z-score is computed. The Risk, Momentum, Quality, Size, and Growth *composites* are then the simple average of their component z-scores (Size and Growth are single-metric, so the "average" is the lone z-score itself). Averaging z-scores (rather than levels) is deliberate: it gives each underlying metric equal weight inside its category and keeps the composites on the same dimensionless scale, which is what Ridge needs.

Size in particular uses `log(marketCap)` rather than raw market cap: raw cap spans six-plus orders of magnitude within a single sector, and within-sector z-scoring a raw level lets the two or three largest names dominate the factor. Log-cap is roughly Gaussian within sector and is the standard Fama-French SMB construction.

## 4. Outlier filter

Rows are dropped if any of `|Risk_Score|`, `|Momentum_Score|`, `|Quality_Score|`, `|Size_Score|`, `|Growth_Score|`, or `|PE_ZScore|` exceeds **3.0** in absolute value (`src/data.py`). The threshold was relaxed from 2.5σ to 3.0σ when the model moved from three predictors to five: the filter is AND'd across all composites plus PE_ZScore (now six conditions, up from four), so each tail bites independently. At 2.5σ that compounds to roughly 6-7% of a typical row being excluded, which pushes the smallest sectors (energy at n ≈ 31 candidates after PE-dropna) to n/p ≈ 5-6 — below the rule-of-thumb n/p ≥ 10 needed for a stable per-sector regression. 3.0σ preserves the universe at ~98% of post-PE-dropna rows while still excluding the most extreme influence points. The cutoff remains hard-coded and is one of the more important single tuning parameters in the pipeline.

## 5. Why Ridge regression

Ridge (L2-regularized linear regression) is used because the five factor composites are correlated by construction (Quality and Risk share leverage and earnings-volatility components; Quality and Growth share margin-expansion components in high-growth names), and because per-sector sample sizes are small (typically 30–150 names after outlier filtering). OLS on five correlated predictors at *n* ≈ 30 produces unstable coefficients whose signs flip across reasonable resamples. Ridge:

- shrinks coefficients toward zero in proportion to their correlation with one another, producing stable, interpretable weights;
- keeps all five factors in the model rather than forcing a selection (Lasso would be the alternative for L1-driven selection; with five predictors and known a-priori-relevant factors, there is little reason to select);
- has a single tuning parameter (`alpha`) which we cross-validate.

The current fit uses `RidgeCV` over the alpha grid `[0.01, 0.1, 1.0, 10.0, 100.0]` with `fit_intercept=False`. Each sector picks its own alpha by k-fold CV on the standardized factor matrix, with `k = min(5, n-1)` so the smallest sectors (energy at n ≈ 30 after outlier filtering) still get a clean CV without exceeding the (n-1) lower bound where k-fold cannot form k disjoint test folds. `fit_intercept=False` is intentional: both inputs and target are z-scores with mean zero within each sector, so the intercept is mechanically zero. Coefficients are taken as absolute values and rescaled to percentages that sum to 100% within the sector; that rescaling is purely for display purposes and is what `weights.csv` reports.

`weights.csv` also persists, per sector, the alpha picked by CV and the in-sample R² of the fitted model. Both are diagnostics — the alpha tells the operator how much shrinkage that sector needed (small alpha → well-conditioned cross-section; large alpha → strongly collinear), and the R² is a "this sector explains X share of within-sector P/E variance" headline. Neither is consumed by the regression itself.

In addition, the refresh script prints a per-sector summary table including the variance inflation factor (VIF) for each of the five factors (`statsmodels.stats.outliers_influence.variance_inflation_factor`). VIF > 10 for any factor in any sector is flagged in the table footer. The flag is informational only: Ridge handles multicollinearity by design through shrinkage, so the factor stays in the model, but the per-sector weight stability for that factor is treated as diagnostic rather than a clean estimate. A walk-forward CV (as opposed to k-fold on a single snapshot) remains the right next step.

## 6. From weights to composite z-score

The per-sector weights are written back into `sector_analysis.csv` as a single number per company:

```
composite_z_score_{s,i} = (
    w_R · Risk_Score
  + w_M · Momentum_Score
  + w_Q · Quality_Score
  + w_S · Size_Score
  + w_G · Growth_Score
) / 100
```

where `w_R + w_M + w_Q + w_S + w_G = 100` within each sector. The dashboard then fits a simple OLS line of `P/E ~ composite_z_score` within each sector (`np.polyfit`, degree 1) and uses that line to produce:

- a **predicted P/E** per company,
- a **deviation** = actual P/E − predicted P/E,
- an **R²** for the sector-level fit.

The deviation is the model's mispricing signal. A positive deviation means the company trades at a higher multiple than its composite z-score predicts versus its sector; a negative deviation means the opposite.

## 7. Interpretation

The deviation is *not* a forecast of forward returns. It is a statement about where a company sits today on a sector-relative valuation surface that the model has fit to today's data. Real mispricing signals require:

1. evidence that today's deviation predicts a measurable forward-return change,
2. survival of that prediction under transaction costs, sector beta, and known factor exposures,
3. a hold-out or walk-forward design that is not curve-fit to the same cross section used to estimate weights.

None of those are in scope here. The deviation is best read as a structured way to ask "why does this company trade where it does versus its peers" — a screen, not a signal.

## 8. Limitations

- **No backtest.** Predicted-vs-actual P/E deviation has not been validated against forward returns. There is no IC, no Sharpe, no holdout. Anyone asking "does this work?" should treat the answer as unanswered.
- **Per-sector sample size.** Sectors with fewer than ~30 surviving names produce noisy Ridge fits even with cross-validated alpha. The outlier filter compounds this; the relaxation from 2.5σ to 3.0σ (§4) was driven by this constraint.
- **R² is heterogeneous.** The cross-sectional fit explains a meaningful share of P/E variance in some sectors and very little in others. The dashboard surfaces R² per sector so this is visible at a glance, and `weights.csv` now persists per-sector R² so the same heterogeneity is visible in the artifact.
- **Point-in-time issues.** Fundamentals are pulled as-of-now from Yahoo Finance with no restatement handling, no proper PIT alignment, and no survivorship-bias correction on the curated ticker list.
- **Growth is single-metric.** yfinance does not populate `epsGrowth` or `cashFlowGrowth`, so the Growth composite reduces to within-sector z-scored revenue growth. A second data vendor would let the composite carry the variety its name suggests.
- **Single data source.** No cross-vendor consistency check. Yahoo Finance field coverage is uneven (`earningsGrowth` is frequently null, banks return `ebitdaMargins=0.0` systematically) and the pipeline mean-imputes or NaN-skips through those gaps.

An empirical backtest of the deviation signal — with all its caveats — is documented in `BACKTEST.md`. The headline numbers there look implausibly strong because the free-tier yfinance data forces several unavoidable shortcuts (notably look-ahead in the composite z-score and the EPS proxy, plus survivorship bias from a today-anchored Russell 1000 universe); `BACKTEST.md` §2 enumerates those constraints and §5 reads the results against them. Anyone asking "does this work?" should read that document before quoting any number.

## 9. What would meaningfully extend this

In rough priority order:

1. **Point-in-time fundamentals** from a paid vendor (Compustat point-in-time, SimFin, S&P Capital IQ). This is the single highest-leverage change: it would invalidate the largest known confounder in the current backtest (look-ahead bias in the composite z-score and the per-ticker EPS proxy) and is a prerequisite for any of the items below being meaningful.
2. **Point-in-time index membership** (CRSP historical Russell 1000 files, or equivalent). At each backtest snapshot, evaluate only the names that were Russell 1000 members on that date. Eliminates the survivorship bias documented in `BACKTEST.md` §2.2.
3. **Longer sample period** — 10-15 years with regime-specific subsamples (2008-2009, 2020-Q1, 2022) — so the backtest can be evaluated under stress rather than only inside the 2023-2026 macro window the current run covers.
4. **Multiple forward-return horizons** (1mo / 3mo / 6mo / 12mo) computed simultaneously. A signal that decays smoothly across horizons is more credible than one that only works at one horizon.
5. **Bootstrap confidence intervals on the Sharpe and IC.** A block-bootstrap with a 3-6 month block length (matched to the autocorrelation structure of monthly equity factor returns) and a 95% CI around the headline numbers.
6. **Walk-forward Ridge refit.** Re-fit the per-sector Ridge weights at each snapshot using only data available at that date (requires (1)). The current pipeline fits once on today's cross-section; a walk-forward fit removes a second layer of look-ahead.
7. A second data vendor for the volatile or missing fields (epsGrowth, cashFlowGrowth, banks' EBITDA margin) so Growth can broaden to a 2-3-metric composite and Quality stops silently dropping a metric for Financials.
8. **Sector-specific cost models** for the backtest. The current 10 bps round-trip is a flat textbook number; replacing it with name-level estimates (or at minimum a quintile-of-ADV-based scaling) would tighten the cost-sensitivity table in `BACKTEST.md` §3.
