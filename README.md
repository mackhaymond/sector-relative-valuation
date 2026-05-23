# sector-relative-valuation

A sector-relative equity valuation model that fits Ridge-weighted factor scores to trailing P/E ratios across 11 GICS sectors and flags companies trading at a meaningful premium or discount to their sector-implied fair value.

## What it does

Comparing P/E ratios across sectors is uninformative. A utility at 18x and a software company at 18x are not "similarly valued" — they sit in industries with structurally different growth, capital intensity, and cost of capital. Any cross-sector ranking on raw multiples picks up sector composition, not mispricing.

This project addresses that by valuing companies *within* their own sector. For each of the 11 GICS sectors it (1) ingests fundamentals and price history for the full Russell 1000 constituent list from Yahoo Finance, (2) reduces five factor groups — Risk, Momentum, Quality, Size, and Growth — to within-sector z-scored composites, and (3) fits a per-sector Ridge regression mapping those composites to the cross-section of trailing P/E ratios. The Ridge alpha is cross-validated per sector. The regression coefficients become sector-specific factor weights, and the resulting predicted P/E acts as a sector-implied fair value benchmark.

Mispricings surface as the deviation between a company's actual P/E and its sector-implied predicted P/E. A Dash/Plotly dashboard surfaces the sector-level fit, the per-company deviation, and a single-stock lookup that scores a user-supplied ticker against its sector peers in real time.

## Methodology

- **Coverage:** the full Russell 1000 (~1,000 stocks, currently 1,002) across 11 GICS sectors (basic-materials, communication-services, consumer-cyclical, consumer-defensive, energy, financial-services, healthcare, industrials, real-estate, technology, utilities). Constituent list lives in `data/russell1000.csv`.
- **Factor universe:** 5 factor categories defined in `src/data.py` — Risk (3 metrics), Momentum (3), Quality (4), Size (1, `log(marketCap)`), Growth (1, `revenueGrowth`). All 5 are wired into the regression.
- **Normalization:** every underlying metric is z-scored *within* its sector before being averaged into a factor composite. This avoids the cross-sector contamination that motivated the project.
- **Outlier handling:** rows are dropped if any of the five factor composites or the P/E z-score exceeds 3.0 standard deviations in absolute value (relaxed from 2.5σ when the model moved from 3 to 5 predictors; see `STRATEGY.md` §4 for the n/p ≥ 10 rationale).
- **Fit:** for each sector, a Ridge regression (`scikit-learn` `RidgeCV` over `[0.01, 0.1, 1.0, 10.0, 100.0]`, no intercept, `k = min(5, n-1)` folds) is fit with the five z-scored factor composites as predictors and the within-sector P/E z-score as the target. Each sector picks its own alpha by CV. Coefficients are taken as the absolute-value normalized factor weights for that sector.
- **Output:** the per-sector weights, chosen alphas, and in-sample R² are persisted to `weights.csv`; a composite z-score (`composite_z_score`) is written back into `sector_analysis.csv`. The dashboard then fits an OLS line of `P/E ~ composite_z_score` within each sector and uses that line to compute a predicted P/E and a deviation per ticker. A per-sector summary table including variance inflation factors (VIF) is printed at the end of the refresh.
- **Methodology detail:** see `STRATEGY.md` for the full rationale (why Ridge, why CV the alpha, why z-scoring within sectors, what the deviation does and does not mean).

## Tech stack

Python 3.12, managed with [uv](https://github.com/astral-sh/uv). Pinned versions from `pyproject.toml`:

- `pandas ^2.2.3`, `numpy ^2.1.3`, `scipy ^1.14.1`, `statsmodels ^0.14.4` — data handling and regression
- `scikit-learn ^1.5.2` — Ridge regression and standardization
- `yfinance ^0.2.49` — fundamentals and price history
- `aiohttp ^3.11.2`, `asyncio` — concurrent data collection with rate limiting
- `dash ^2.18.2`, `plotly ^5.24.1` — interactive dashboard
- `tqdm ^4.67.0` — progress reporting

Containerized via a multi-stage `Dockerfile`. Data refresh runs in GitHub Actions; deployment is orchestrated via Nomad.

## Repository structure

```
.
├── src/
│   ├── data.py              # Async collection of 5 factor groups from Yahoo Finance
│   │                        # plus within-sector z-scoring and 3.0σ outlier filter
│   ├── generate_weights.py  # Per-sector Ridge regression and composite z-score writeback
│   └── dashboard.py         # Dash/Plotly UI: sector view, single-stock lookup, factor selector
├── sector_analysis.csv      # Simplified per-ticker output (sector, composites, PE, composite_z_score)
├── sector_analysis_full.csv # Full per-ticker output with every collected metric and z-score
├── weights.csv              # Per-sector Ridge-derived factor weights
├── Dockerfile               # Multi-stage build, non-root runtime
├── jobspec.nomad.hcl        # Nomad deployment spec
├── pyproject.toml           # uv-managed dependencies and project metadata
├── uv.lock                  # locked dependency graph
└── .github/workflows/       # CI: Docker publish + scheduled data refresh
```

## Running locally

Prerequisites: Python 3.12, [uv](https://github.com/astral-sh/uv).

```sh
uv sync
uv run python src/data.py              # refresh the dataset (rate-limited Yahoo calls; takes minutes)
uv run python src/generate_weights.py  # fit per-sector Ridge weights
uv run python src/dashboard.py         # serve dashboard on http://localhost:8050
```

To run the prebuilt container:

```sh
docker build -t sector-relative-valuation .
docker run -p 8050:8050 sector-relative-valuation
```

The repo ships with a recent `sector_analysis.csv` and `weights.csv` so the dashboard can be served without refreshing the dataset first.

## Limitations and known gaps

- **No formal backtest.** The model is fit cross-sectionally on a single snapshot of fundamentals; predicted-vs-actual P/E deviation has not been validated as a forward-return signal. There is no holdout, walk-forward, or transaction-cost analysis. Treat all output as descriptive, not prescriptive.
- **Coverage is the Russell 1000.** ~1,000 large- and mid-cap US-listed names across 11 sectors. Conclusions do not generalize to small caps, micro caps, or international names that are not Russell constituents.
- **R² varies materially by sector.** The within-sector linear fit between composite z-score and P/E is informative in some sectors and weak in others. The dashboard displays the R² per sector so this is visible at a glance rather than buried.
- **Growth is single-metric.** yfinance does not populate `epsGrowth` or `cashFlowGrowth`; the Growth composite reduces to within-sector z-scored revenue growth. An earlier iteration collected a separate Profitability category (folded into Quality via EBITDA margin instead), Value (mechanically circular with the trailing-P/E target), and Liquidity (weak P/E predictor that breaks for financials); all three were dropped from the pipeline.
- **Single data source.** Yahoo Finance is the only feed; fundamentals are point-in-time as scraped, with no PIT correction or restatement handling.
- **No real-money usage.** This is a research framework, not a trading system.

## Background

The repository was started by Ezra Schwartz; the framework design, the async data pipeline, the per-sector Ridge approach, and the dashboard are by Mack Haymond. A later extension of the model by Alex Sod was used in an external quantitative-finance club competition; that extension is not the focus of this repository.

## License

MIT. See `LICENSE`.
