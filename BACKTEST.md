# Backtest

An empirical test of whether the per-ticker deviation signal produced by the sector-relative valuation model (see `STRATEGY.md`) predicts forward stock returns.

This document reports two runs of the same backtest design:

- **Phase 3** (initial) — used yfinance's current-snapshot fundamentals at every historical date. Inflated by look-ahead bias.
- **Phase 3.5** (current) — uses SimFin point-in-time fundamentals (filtered by `Publish Date <= as_of`) and refits the per-sector Ridge weights at each snapshot (walk-forward). Removes the look-ahead bias.

The Phase 3 numbers are preserved alongside the Phase 3.5 numbers in §3 so the diff is visible — that diff is the credibility of the correction.

**Headline (36-month run, 10 bps round-trip cost, Phase 3.5 PIT pipeline):**

| Metric                    | Phase 3 (look-ahead) | Phase 3.5 (PIT)    |
| ------------------------- | -------------------- | ------------------ |
| Mean Spearman IC          | -0.0799              | **-0.0128**        |
| IC t-stat                 | -8.85                | **-1.17**          |
| IC information ratio      | -2.98                | **-0.40**          |
| LS annualized Sharpe      | 3.36                 | **0.054**          |
| LS cumulative return      | +93.96%              | **+0.43%**         |
| LS max drawdown           | -1.72%               | **-12.02%**        |
| LS hit rate (months > 0)  | 80.56%               | **38.89%**         |

A negative IC is the "signal works" direction. The Phase 3.5 IC of -0.0128 is below the published reference range (0.02-0.05 |IC| for working cross-sectional alpha factors) and is not statistically significant (|t| = 1.17, below the conventional 2.0 cutoff). The cost-sensitivity table at the bottom of §3 shows the long-short return goes negative at 25 bps. **Read as a null result.** See §5 for the honest interpretation.

## 1. Methodology

### Universe and snapshot schedule

- **Universe:** the 795 tickers carried by `sector_analysis.csv`. Sourced from `data/russell1000.csv` (today's Russell 1000 constituent list). The universe is fixed across the backtest window; per-snapshot per-sector survival requirements (≥ 15 tickers after the PIT-pipeline 3.0σ outlier filter) decide which sectors participate at each snapshot.
- **Sample window:** trailing 36 calendar months ending at the most recent available trading day (configurable via `--months`).
- **Snapshot frequency:** monthly. Each snapshot is the first available trading day on or after a calendar month-start. The last usable snapshot is constrained so its t+1 month forward date is still inside the fetched price index.

### Signal construction (Phase 3.5 PIT pipeline)

At each snapshot date `t`, for each ticker in the universe:

1. **Fetch PIT fundamentals** via `src/pit_fundamentals.fetch_pit_metrics(ticker, t)`. This pulls SimFin's TTM income + balance datasets (bulk-cached) and selects the most-recent filing whose `Publish Date <= t`. Returns TTM EPS, revenue, ROE, ROA, operating margin, EBITDA margin, debt/equity, revenue growth, earnings growth, shares outstanding. **`Publish Date`, not `Report Date`, is the as-of cutoff** — a 2023-Q4 filing has a `Report Date` of 2023-12-31 but doesn't become public until February 2024; using `Report Date` would leak two months of look-ahead.
2. **Derive price metrics** from the trailing 1y of the cached yfinance price series ending at `t`: `MaxDrawdown`, `ReturnSD`, `RSI`, `PriceChange12M`, `price_at_t`. These are PIT-correct because they use only prices on dates ≤ `t`.
3. **Derive market cap** at `t`: `MarketCap = SharesOutstanding × price_at_t`, then `LogMarketCap = ln(MarketCap)`.
4. **Compute regression target**: `PE_t = price_at_t / TTM_EPS_t` (both are PIT).
5. **Within-sector pipeline (per snapshot, per sector):**
   - Z-score each of the 12 raw metrics within the sector (mean-imputed for missing values, matching `src/data.py:404`).
   - Compose 5 factor scores by averaging the relevant z-scores: `Risk_Score`, `Momentum_Score`, `Quality_Score`, `Size_Score`, `Growth_Score`.
   - Apply the 3.0σ outlier filter on the 5 composites + `PE_ZScore` (matches `src/data.py:426-443`).
   - **Fit `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], fit_intercept=False, cv=max(2, min(5, n-1)))`** on (standardized 5 composites → `PE_ZScore`). This produces snapshot-specific factor weights — the walk-forward correction that eliminates the Phase-3 "today-fitted weights leak across all snapshots" look-ahead.
   - Normalize the coefficients to sum to 100 (matching `src/generate_weights.py:77`).
6. **Per-ticker composite z-score:** `composite_z_score_i = (Σ w_k · k_score_i) / 100`.
7. **Within-sector 1D OLS** of `PE_t` on `composite_z_score` (matches the dashboard's scatter fit at `src/dashboard.py:500`). Predicted PE and deviation: `predicted_PE_i = slope · composite_z_score_i + intercept`, `deviation_i = PE_t_i - predicted_PE_i`.

A more negative deviation = "cheaper than the model predicts" = long candidate. More positive = "richer than the model predicts" = short candidate.

### Portfolio construction

Within each sector at each snapshot:
- Surviving tickers are ranked by `deviation`.
- 5 quintiles formed (`pandas.qcut`, `duplicates='drop'`).
- **Long leg:** Q1 (most-negative). **Short leg:** Q5 (most-positive). Equal-weight within each leg.
- Sectors with fewer than 15 surviving tickers after the PIT pipeline are skipped.

Across sectors at each snapshot:
- Per-sector LS returns are averaged with **equal sector weights** (sector neutralization).

### Forward returns and costs

`fwd_return_i = (price_{t+1mo,i} - price_t,i) / price_t,i` using split- and dividend-adjusted prices. Tickers without a t+1mo price (delisted between rebalances) are dropped.

Round-trip transaction cost defaults to 10 bps per snapshot, deducted from each month's gross long-short return. The strategy is treated as fully turning over each month (conservative; some names persist in practice).

### Reproducibility

```sh
# Set the SimFin API key (free tier — sign up at https://www.simfin.com/en/)
# Add to .env:  SIMFIN_API_KEY=your_key_here  (see .env.example)

uv run python src/backtest.py --months 36 --cost-bps 10
```

Outputs: `backtest_results.csv` (one row per (snapshot, sector)), `backtest_artifacts/cumulative_long_short_return.png`, `backtest_artifacts/ic_distribution_by_sector.png`. The yfinance price cache is at `backtest_cache/prices.pkl`; the SimFin bulk-downloaded TTM income + balance datasets live at `backtest_cache/simfin/` (both gitignored; regenerable).

## 2. Methodology limitations

### 2.1 [HISTORICAL] Fundamentals look-ahead — fixed in Phase 3.5

The Phase 3 backtest scored the signal at each historical snapshot using `composite_z_score` and `PE` columns from `sector_analysis.csv`. Those values are *as of the latest data refresh*. Using today's fundamentals to value yesterday's prices is look-ahead bias: at `t = 2023-05`, the signal could "see" today's revenue growth, today's RSI, today's leverage. **Phase 3.5 fixes this** by fetching SimFin's filings with `Publish Date <= t` and refitting the Ridge weights per snapshot. The historical Phase 3 column in §3's comparison table was generated under this caveat and should NOT be interpreted as out-of-sample.

### 2.2 Survivorship bias (now the dominant caveat)

`data/russell1000.csv` is *today's* Russell 1000 constituent list. Companies that were Russell 1000 members in 2023 but have since been delisted, merged, or demoted to the Russell 2000 do not appear in the universe at any snapshot. The backtest's universe at `t = 2023-05` therefore systematically excludes the names that subsequently failed, which mechanically inflates every aggregate return metric. **Correcting this requires a point-in-time index membership feed (CRSP, FTSE-Russell historical)** — out of scope for the free-data tier.

### 2.3 SimFin Financials coverage gap

SimFin's free-tier balance-sheet TTM dataset uses a schema for banks and insurers that doesn't carry `Total Equity`, `Long Term Debt`, etc. in the same column names as non-financials. Most Financials tickers therefore return all-None from `fetch_pit_metrics` and fall out of the universe at the snapshot. In the Phase 3.5 run, the financial-services sector still appears (36 snapshots) because some non-bank financials are covered, but the per-sector n is lower than under the Phase 3 yfinance path. **Two sectors — communication-services (13 snapshots) and utilities (17 snapshots) — fall below the conventional snapshot count** because of a combination of SimFin coverage and the 3.0σ filter at smaller sector sizes.

### 2.4 Transaction-cost assumption

A flat 10 bps round-trip cost is a textbook approximation, not a measured number. Actual costs vary materially with liquidity, borrow availability, and market impact. The cost-sensitivity table at the bottom of §3 reports Sharpe and cumulative return at 0 / 10 / 25 bps; at 25 bps the Phase 3.5 LS Sharpe goes **negative** (-0.21) — the signal does not survive even modest cost stress.

### 2.5 Single forward-return horizon

The forward-return horizon is locked at 1 month and is NOT configurable from the CLI — a deliberate anti-p-hacking guard. A signal that "works" only at one horizon is suspicious; testing multiple horizons should be done with bootstrap CIs (§6).

### 2.6 Sample-period dependence

36 months covers May 2023 through May 2026 — a single macro regime, dominated by AI capex, post-pandemic rate normalization, and tech-led indices. Conclusions from this window do not generalize across regimes.

### 2.7 Historical sector classification

`data/russell1000.csv` carries today's GICS sector per ticker. The backtest reuses that sector across all snapshots; the residual error from intra-window sector reclassifications is expected to be small (sector reclassifications are rare over 3-year windows) but is not zero.

## 3. Results

### Headline comparison (36 months, 10 bps round-trip cost)

| Metric                       | Phase 3 (look-ahead) | Phase 3.5 (PIT) | Diff |
| ---------------------------- | -------------------: | --------------: | ---: |
| Months                       | 36                   | 36              | — |
| Snapshots                    | 36                   | 36              | — |
| Cost (bps round-trip)        | 10.00                | 10.00           | — |
| **Mean IC (Spearman)**       | -0.0799              | **-0.0128**     | 84% smaller |
| **IC t-stat**                | -8.85                | **-1.17**       | not significant |
| **IC information ratio**     | -2.98                | **-0.40**       | — |
| LS mean monthly return (net) | 0.01875              | 0.00031         | -98% |
| LS monthly std (net)         | 0.01933              | 0.02002         | similar |
| **LS annualized Sharpe**     | 3.36                 | **0.054**       | -98% |
| **LS cumulative return**     | +93.96%              | **+0.43%**      | -99% |
| LS max drawdown              | -1.72%               | **-12.02%**     | 7x worse |
| LS hit rate (months > 0)     | 80.56%               | **38.89%**      | below 50% |

The Phase 3.5 column is the credible result. Mean IC of -0.0128 with |t| = 1.17 is **not statistically distinguishable from zero**. The strategy returns essentially 0% over 36 months net of cost, with a 12% peak-to-trough drawdown and a sub-50% hit rate. **The deviation signal does not predict forward monthly returns under PIT conditions.**

### Cost sensitivity (Phase 3.5)

| cost_bps | monthly_ret | Sharpe (ann) | cum_ret | hit_rate |
| -------: | ----------: | -----------: | ------: | -------: |
|     0.00 |     0.00131 |        0.227 | +4.11%  |   41.67% |
|    10.00 |     0.00031 |        0.054 | +0.43%  |   38.89% |
|    25.00 |    -0.00119 |       -0.205 | -4.85%  |   36.11% |

Even at zero cost the Phase 3.5 Sharpe is 0.23 — a magnitude that vanishes once realistic frictions are added. At 25 bps the strategy loses money. This is the opposite of the Phase 3 cost-sensitivity profile (where Sharpe stayed above 3.0 across all three cost levels) and is what an honest null-result looks like.

### Cumulative long-short return (Phase 3.5)

![Cumulative long-short return](backtest_artifacts/cumulative_long_short_return.png)

A flat curve with a sustained ~12% drawdown in the middle of the window and recovery to near-flat by the end. Compare to the Phase 3 chart in the git history (commit `9fbdc3c`): a near-monotone climb to +94% with virtually no drawdown — the visual signature of look-ahead bias.

### IC distribution by sector (Phase 3.5)

![IC distribution by sector](backtest_artifacts/ic_distribution_by_sector.png)

Per-snapshot IC distributions now straddle zero in most sectors. Five of the eleven sectors have positive median IC (i.e., the signal is wrong-directional) under PIT conditions — consistent with the headline mean IC being indistinguishable from zero.

## 4. Per-sector breakdown (Phase 3.5)

| Sector                  | n_snaps | mean IC | IC t-stat | LS mean monthly | LS Sharpe | LS hit rate |
| ----------------------- | ------: | ------: | --------: | --------------: | --------: | ----------: |
| basic-materials         |      36 | +0.0433 |     +1.40 |         -0.0038 |    -0.305 |      50.00% |
| communication-services  |      13 | -0.1596 |     -3.33 |          0.0361 |     2.241 |      69.23% |
| consumer-cyclical       |      36 | -0.0449 |     -1.77 |          0.0088 |     0.816 |      52.78% |
| consumer-defensive      |      36 | +0.0789 |     +1.93 |         -0.0173 |    -1.250 |      36.11% |
| energy                  |      36 | +0.0161 |     +0.32 |         -0.0076 |    -0.513 |      36.11% |
| financial-services      |      36 | -0.0713 |     -1.93 |          0.0085 |     0.727 |      63.89% |
| healthcare              |      36 | -0.0474 |     -1.65 |          0.0094 |     0.678 |      50.00% |
| industrials             |      36 | +0.0025 |     +0.10 |         -0.0039 |    -0.360 |      52.78% |
| real-estate             |      36 | -0.0361 |     -1.20 |          0.0027 |     0.317 |      47.22% |
| technology              |      36 | -0.0328 |     -1.18 |          0.0041 |     0.303 |      52.78% |
| utilities               |      17 | +0.0497 |     +1.02 |          0.0046 |     0.350 |      47.06% |

**Sector-level signal heterogeneity is now visible:**
- The only sector with statistically significant signal at |t| > 2 is communication-services (t = -3.33), but it has only 13 snapshots — small-N caveat applies, and the signal may be noise dressed up by limited sample size.
- Five sectors have **positive** mean IC (wrong-directional): basic-materials, consumer-defensive, energy, industrials, utilities. In those sectors a "cheap" deviation is associated with *lower* subsequent returns over this window.
- Six sectors have negative mean IC (right-directional) but only one (communication-services) is significant.
- Per-sector long-short Sharpes are split: 6 sectors positive, 5 sectors negative. Average across all sectors with equal weight is the headline 0.054 — almost exactly zero, consistent with the cross-sector ICs largely cancelling.

This matches `STRATEGY.md` §8's earlier observation that the within-sector R² of the fit is heterogeneous across sectors. The Phase 3.5 backtest now shows that **the heterogeneity is not a tail of "strong" sectors — it's signal scattering in both directions.**

## 5. Honest interpretation

**The deviation signal does not predict 1-month forward returns under point-in-time conditions on the Russell 1000 over May 2023 — May 2026.** Mean IC -0.0128 (|t| = 1.17, not significant), annualized Sharpe 0.054 net of 10 bps cost (turns negative at 25 bps), cumulative net return +0.43% over 36 months, max drawdown -12%, hit rate 38.9%. The Phase 3 backtest reported a Sharpe of 3.36 and a +94% cumulative return for the identical strategy design — that result was almost entirely look-ahead bias from using today's composite z-score and today's per-ticker EPS at every historical snapshot. After replacing yfinance's current-snapshot fundamentals with SimFin point-in-time fundamentals (filtered by `Publish Date <= as_of`) and refitting the per-sector Ridge weights at each snapshot, the apparent signal collapsed by ~98% on every return metric. The plumbing was correct; the signal was not.

That doesn't mean the model is worthless — it means its **as-of-today** cross-sectional output (the dashboard's predicted-PE-vs-actual-PE deviation) is a structured way to describe relative valuation, not a forecast. It's a screen, not a signal. The two outstanding methodology gaps that could move the result are: (1) point-in-time index membership (survivorship — §2.2), which would let the universe at `t = 2023-05` include the names that subsequently delisted, possibly raising or lowering the headline depending on which side of the long-short they would have populated; and (2) a longer sample window that spans multiple regimes (§2.6), which would test whether the near-zero result is regime-specific or general. Both require paid data feeds — see §6.

**One-line interview answer:** "Phase 3 of my backtest reported a Sharpe of 3.4. The correct answer was 0.05. The diff was look-ahead bias in the fundamentals — fixed with point-in-time data from SimFin and walk-forward Ridge refit. The deviation signal does not predict forward returns at monthly horizon over this window. I document both numbers so the methodology fix is visible."

## 6. What a more rigorous test would look like

In rough priority order — items 1 and 2 are now the highest-leverage open work; (3)-(8) are pointless without them.

1. **Point-in-time index membership** (CRSP / FTSE-Russell historical). At each snapshot, evaluate only the names that were Russell 1000 members on that date. Removes survivorship bias (§2.2).
2. **Longer sample period** — 10-15 years with regime-specific subsamples (2008-2009, 2018-Q4, 2020-Q1, 2022). The Phase 3.5 36-month window covers a single benign-for-mean-reversion macro regime; a regime-spanning window would test whether the near-zero IC is sample-specific or general.
3. **Cross-vendor PIT fundamentals.** SimFin's free tier has the Financials coverage gap (§2.3) and updates infrequently. A second feed (Compustat point-in-time, S&P Capital IQ) would let Financials rejoin the universe and tighten any per-sector signal estimate.
4. **Multiple forward-return horizons** (1mo / 3mo / 6mo / 12mo) computed simultaneously. The 1mo horizon is locked here to prevent p-hacking; a principled multi-horizon test reported jointly with bootstrap CIs would let an analyst see if signal accumulates at longer horizons.
5. **Bootstrap confidence intervals on the Sharpe and IC.** Block-bootstrap the monthly LS return series with a 3-6 month block (matched to monthly equity-factor autocorrelation). Even the Phase 3.5 0.054 Sharpe has wide CIs at n=36 monthly observations.
6. **Sector-specific cost models.** Replace the flat 10 bps with name-level estimates (Trade Cost Analysis from any execution provider) or a quintile-of-ADV-based scaling.
7. **Borrow-availability filter on the short leg.** Drop hard-to-borrow names.
8. **Walk-forward Ridge refit alpha grid.** Phase 3.5 reuses Phase 2's `[0.01, 0.1, 1.0, 10.0, 100.0]`. A finer grid or a different alpha at each snapshot could change the weights modestly; the effect on aggregate Sharpe is likely second-order, but worth quantifying.
