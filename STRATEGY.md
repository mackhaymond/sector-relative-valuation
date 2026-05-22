# Strategy and Methodology

This document describes how the model is constructed and the reasoning behind each design choice. It is intentionally specific so that a reader who wants to challenge the methodology can locate the assumption they want to push back on.

## 1. Why sector-relative valuation

The headline problem this project addresses is that cross-sector comparisons of trailing P/E ratios are not meaningful. Sectors differ structurally in:

- **Capital intensity and depreciation schedules** — utilities and energy carry meaningfully different earnings volatility and reinvestment burdens than asset-light software.
- **Growth expectations baked into multiples** — communication services and technology typically trade on forward growth, real estate and consumer staples on yield and cash conversion.
- **Cost of capital** — financials' multiples are sensitive to rate regimes in a way that consumer cyclical multiples are not.

Ranking AAPL against XOM on P/E ranks Apple's sector against ExxonMobil's sector, not the two companies. To compare companies on valuation, the unit of comparison has to be *within* the sector — the company versus its own peers. Every step in the pipeline is built around that constraint.

## 2. Factor universe

Eight factor categories are collected in `src/data.py`, each composed of three underlying metrics (24 metrics total):

| Category        | Metrics                                                                |
| --------------- | ---------------------------------------------------------------------- |
| Risk            | Max drawdown (1y), debt-to-equity, return standard deviation (1y)      |
| Momentum        | 52-week price change, RSI, earnings growth                             |
| Quality         | ROE, ROA, operating margin                                             |
| Value           | Price-to-book, EV/EBITDA, price-to-sales                               |
| Size            | Market cap, total assets, enterprise value                             |
| Growth          | Revenue growth, EPS growth, cash flow growth                           |
| Profitability   | Gross margin, EBITDA margin, net profit margin                         |
| Liquidity       | Current ratio, quick ratio, interest coverage                          |

Three categories — Risk, Momentum, Quality — are weighted in the production fit. The remaining five are collected and surfaced in the dashboard's Factor Selection tab as a scaffold for future model iterations, but are not yet incorporated into the Ridge regression that produces sector weights. This is an honest gap, not an oversight; widening the model has design implications (multicollinearity between Value and Quality, between Size and Liquidity) that are not addressed here.

The target variable is the trailing P/E ratio. The framework supports alternative valuation targets (price-to-book, price-to-sales, price-to-free-cashflow) by changing the `Y_VALUATION_METRIC` map; only trailing P/E is exercised in the current fit.

## 3. Within-sector z-scoring

Every metric is standardized within its sector before being combined. For metric *m* in sector *s*:

```
z_{m,s,i} = (x_{m,s,i} - mean_{m,s}) / std_{m,s}
```

Missing values are mean-imputed within the sector before the z-score is computed. The Risk, Momentum, and Quality *composites* are then the simple average of their three component z-scores. Averaging z-scores (rather than levels) is deliberate: it gives each underlying metric equal weight inside its category and keeps the composites on the same dimensionless scale, which is what Ridge needs.

## 4. Outlier filter

Rows are dropped if any of `|Risk_Score|`, `|Momentum_Score|`, `|Quality_Score|`, or `|PE_ZScore|` exceeds 2.5 in absolute value (`src/data.py`). The threshold is chosen conservatively — at a 2.5σ cutoff on roughly-normal scores roughly 1.2% of observations are excluded per tail, which removes the worst influence points without gutting small-sector samples. The cutoff is hard-coded and is the most important single tuning parameter in the pipeline.

## 5. Why Ridge regression

Ridge (L2-regularized linear regression) is used because the three factor composites are correlated by construction (Quality and Risk share leverage and earnings-volatility components), and because per-sector sample sizes are small (typically 30–45 names after outlier filtering). OLS on three correlated predictors at *n* ≈ 30 produces unstable coefficients whose signs flip across reasonable resamples. Ridge:

- shrinks coefficients toward zero in proportion to their correlation with one another, producing stable, interpretable weights;
- keeps all three factors in the model rather than forcing a selection (Lasso would be the alternative for L1-driven selection, but with only three predictors there is little reason to select);
- has a single tuning parameter (`alpha`) that admits a sensible default.

The current fit uses `alpha=1.0` and `fit_intercept=False`. `fit_intercept=False` is intentional: both the inputs and the target are z-scores with mean zero within each sector, so the intercept is mechanically zero. Coefficients are taken as absolute values and rescaled to percentages that sum to 100% within the sector; that rescaling is purely for display purposes and is what `weights.csv` reports.

`alpha` has not been tuned via cross-validation. A walk-forward CV would be the right next step but is not implemented.

## 6. From weights to composite z-score

The per-sector weights are written back into `sector_analysis.csv` as a single number per company:

```
composite_z_score_{s,i} = (w_R · Risk_Score + w_M · Momentum_Score + w_Q · Quality_Score) / 100
```

where `w_R + w_M + w_Q = 100` within each sector. The dashboard then fits a simple OLS line of `P/E ~ composite_z_score` within each sector (`np.polyfit`, degree 1) and uses that line to produce:

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
- **Per-sector sample size.** Sectors with fewer than ~30 surviving names produce noisy Ridge fits. The outlier filter compounds this.
- **R² is heterogeneous.** The cross-sectional fit explains a meaningful share of P/E variance in some sectors and very little in others. The dashboard surfaces R² per sector so this is visible at a glance.
- **Point-in-time issues.** Fundamentals are pulled as-of-now from Yahoo Finance with no restatement handling, no proper PIT alignment, and no survivorship-bias correction on the curated ticker list.
- **Three factors out of eight.** Value, Size, Growth, Profitability, Liquidity are collected and stored but not weighted. The model implicitly attributes variance from those factors to the three that are included; correlated omitted variables will bias the included coefficients.
- **Single data source.** No cross-vendor consistency check. Yahoo Finance field coverage is uneven (e.g. `earningsGrowth` is frequently null) and the pipeline mean-imputes through those gaps.

## 9. What would meaningfully extend this

In rough priority order:

1. A walk-forward backtest of deviation-versus-forward-return, broken out by sector.
2. Cross-validated `alpha` selection per sector, with the chosen `alpha` reported alongside the weights.
3. Widening the factor set to all eight categories, with multicollinearity diagnostics (VIF) and a defensible rule for when to drop or merge categories.
4. A second data vendor for the most volatile fields (earnings growth, ROE) to reduce noise from Yahoo Finance imputation.
5. A proper PIT framework so the same fit can be re-run on historical dates.
