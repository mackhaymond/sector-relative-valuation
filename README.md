# sector-relative-valuation

A sector-relative equity valuation model that fits Ridge-weighted factor scores to trailing P/E ratios across 11 GICS sectors and flags companies trading at a meaningful premium or discount to their sector-implied fair value.

## What it does

Comparing P/E ratios across sectors is uninformative. A utility at 18x and a software company at 18x are not "similarly valued" — they sit in industries with structurally different growth, capital intensity, and cost of capital. Any cross-sector ranking on raw multiples picks up sector composition, not mispricing.

This project addresses that by valuing companies *within* their own sector. For each of the 11 GICS sectors it (1) ingests fundamentals and price history for a curated sample of constituent stocks from Yahoo Finance, (2) reduces three factor groups — Risk, Momentum, and Quality — to within-sector z-scored composites, and (3) fits a per-sector Ridge regression mapping those composites to the cross-section of trailing P/E ratios. The regression coefficients become sector-specific factor weights, and the resulting predicted P/E acts as a sector-implied fair value benchmark.

Mispricings surface as the deviation between a company's actual P/E and its sector-implied predicted P/E. A Dash/Plotly dashboard surfaces the sector-level fit, the per-company deviation, and a single-stock lookup that scores a user-supplied ticker against its sector peers in real time.

## Methodology

- **Coverage:** ~470 stocks across 11 GICS sectors (basic-materials, communication-services, consumer-cyclical, consumer-defensive, energy, financial-services, healthcare, industrials, real-estate, technology, utilities).
- **Factor universe:** 8 factor categories defined in `src/data.py` (Risk, Momentum, Quality, Value, Size, Growth, Profitability, Liquidity), each composed of 3 underlying metrics — 24 metrics total are collected.
- **Current model:** 3 of the 8 factor groups (Risk, Momentum, Quality) are wired into the regression. The remaining 5 are collected and exposed in the dashboard's Factor Selection tab as a scaffold for future iterations; they are not yet weighted in the production fit.
- **Normalization:** every underlying metric is z-scored *within* its sector before being averaged into a factor composite. This avoids the cross-sector contamination that motivated the project.
- **Outlier handling:** rows are dropped if any of the three factor composites or the P/E z-score exceeds 2.5 standard deviations in absolute value.
- **Fit:** for each sector, a Ridge regression (`scikit-learn`, `alpha=1.0`, no intercept) is fit with the three z-scored factor composites as predictors and the within-sector P/E z-score as the target. Coefficients are taken as the absolute-value normalized factor weights for that sector.
- **Output:** the per-sector weights are persisted to `weights.csv` and a composite z-score (`composite_z_score`) is written back into `sector_analysis.csv`. The dashboard then fits an OLS line of `P/E ~ composite_z_score` within each sector and uses that line to compute a predicted P/E and a deviation per ticker.
- **Methodology detail:** see `STRATEGY.md` for the full rationale (why Ridge, why z-scoring within sectors, what the deviation does and does not mean).

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
│   ├── data.py              # Async collection of all 8 factor groups from Yahoo Finance
│   │                        # plus within-sector z-scoring and 2.5σ outlier filter
│   ├── generate_weights.py  # Per-sector Ridge regression and composite z-score writeback
│   └── dashboard.py         # Dash/Plotly UI: sector view, single-stock lookup, factor selector
├── sector_analysis.csv      # Simplified per-ticker output (sector, composites, PE, composite_z_score)
├── sector_analysis_full.csv # Full per-ticker output with every collected metric and z-score
├── weights.csv              # Per-sector Ridge-derived factor weights
├── Dockerfile               # Multi-stage build, non-root runtime
├── jobspec.nomad.hcl        # Nomad deployment spec
├── pyproject.toml           # Poetry-managed dependencies
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
- **Coverage is curated.** ~470 tickers across 11 sectors — a hand-picked sample of large constituents, not the full sector membership. Conclusions do not generalize to small caps or international names not in the list.
- **R² varies materially by sector.** The within-sector linear fit between composite z-score and P/E is informative in some sectors and weak in others. The dashboard displays the R² per sector so this is visible at a glance rather than buried.
- **Only 3 of 8 factor groups are weighted.** Value, Size, Growth, Profitability, and Liquidity are collected and exposed in the UI but not yet incorporated into the Ridge fit.
- **Single data source.** Yahoo Finance is the only feed; fundamentals are point-in-time as scraped, with no PIT correction or restatement handling.
- **No real-money usage.** This is a research framework, not a trading system.

## Background

The repository was started by Ezra Schwartz; the framework design, the async data pipeline, the per-sector Ridge approach, and the dashboard are by Mack Haymond. A later extension of the model by Alex Sod was used in an external quantitative-finance club competition; that extension is not the focus of this repository.

## License

MIT. See `LICENSE`.
