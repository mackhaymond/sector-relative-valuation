import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from tqdm import tqdm
import time
import asyncio
import aiohttp
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

#######################
# MASTER VARIABLES
#######################

# Target Sectors for Analysis
SECTORS = [
    "basic-materials",
    "communication-services",
    "consumer-cyclical",
    "consumer-defensive",
    "energy",
    "financial-services",
    "healthcare",
    "industrials",
    "real-estate",
    "technology",
    "utilities"
]

# Independent Variables (X) for Regression

# Category 1: Risk Metrics - Measures of company's risk profile
X1_RISK_METRICS = {
    "MaxDrawdown": "maxDrawdown",           # Maximum drawdown over last year
    "DebtToEquity": "debtToEquity",         # Leverage ratio
    "ReturnSD": "returnSD"                   # Standard deviation of returns
}

# Category 2: Momentum Metrics - Measures of price momentum and growth
X2_MOMENTUM_METRICS = {
    "PriceChange12M": "52WeekChange",     # 12-month price change
    "RSI": "rsi",                           # Relative Strength Index
    "EarningsGrowth": "earningsGrowth"      # Earnings growth
}

# Category 3: Quality Metrics - Measures of company's operational efficiency
#
# EBITDAMargin uses the plural yfinance field name ``ebitdaMargins``;
# the singular form (which an earlier iteration of this codebase
# collected) is not present on the info dict and returned 100% null.
# Banks return ebitdaMargins=0.0 because their cost-of-revenue concept
# (interest expense) doesn't fit the standard EBITDA definition. That
# distorts the raw value but not the within-sector z-score: if every
# financial returns 0, the metric std is 0 and the z-score is NaN,
# which mean(axis=1) skips, so Financials' Quality_Score becomes the
# mean of the other three metrics. The composite is robust by accident.
X3_QUALITY_METRICS = {
    "ROE": "returnOnEquity",                # Return on Equity
    "ROA": "returnOnAssets",                # Return on Assets
    "OperatingMargin": "operatingMargins",  # Operating efficiency
    "EBITDAMargin": "ebitdaMargins",        # EBITDA / Revenue
}

# Category 5: Size Metrics
#
# Single-metric composite: ``log(marketCap)``. Raw market cap spans 6+
# orders of magnitude inside a single sector; z-scoring the level lets
# the 2-3 largest names dominate the factor. Log-cap is roughly Gaussian
# within sector and is the standard SMB construction. EnterpriseValue
# and TotalAssets (the other two original X5 fields) were 0.95+
# correlated with marketCap or 100% null, so neither contributed
# meaningful independent information; both were dropped.
X5_SIZE_METRICS = {
    "LogMarketCap": "logMarketCap",         # ln(marketCap); derived in get_metric_async
}

# Category 6: Growth Metrics
#
# yfinance does not expose epsGrowth or cashFlowGrowth on the info
# dict; both came back 100% null across the Russell 1000 (verified
# against sector_analysis_full.csv after the Phase 1 refresh).
# revenueGrowth is reliably populated (~99.6%). Until a second data
# source is wired in, Growth is a single-metric composite.
X6_GROWTH_METRICS = {
    "RevenueGrowth": "revenueGrowth",       # Revenue Growth
}

# Profitability is intentionally not a standalone factor: Quality already
# captures the same income-statement signal (ROE, ROA, OperatingMargin,
# EBITDAMargin in a follow-up commit). Treating Quality and Profitability
# as independent factors would double-count.

# X4, X7, X8 identifiers are intentionally absent. The category numbers
# are stable across the codebase; renumbering the survivors would
# silently rewrite csv column orderings and dashboard wiring.

# Dependent Variable (Y) for Regression
Y_VALUATION_METRIC = {
    "PE": "trailingPE"  # Trailing P/E ratio as valuation metric
    # Can be modified to use different valuation metrics:
    # "PB": "priceToBook"
    # "PS": "priceToSales"
    # "PFCF": "priceToFreeCashflow"
}

# Combine all metrics for data collection
ALL_METRICS = {
    **X1_RISK_METRICS,
    **X2_MOMENTUM_METRICS,
    **X3_QUALITY_METRICS,
    **X5_SIZE_METRICS,
    **X6_GROWTH_METRICS,
    **Y_VALUATION_METRIC
}

METRICS = {
    "x1_risk_metrics": X1_RISK_METRICS,
    "x2_momentum_metrics": X2_MOMENTUM_METRICS,
    "x3_quality_metrics": X3_QUALITY_METRICS,
    "x5_size_metrics": X5_SIZE_METRICS,
    "x6_growth_metrics": X6_GROWTH_METRICS,
    "y_valuation_metric": Y_VALUATION_METRIC,
    "all_metrics": ALL_METRICS,
}

# API Settings
MAX_CONCURRENT_REQUESTS = 2
REQUEST_DELAY = 0.5  # seconds
INDUSTRY_DELAY = 1.0  # seconds

#######################
# LOGGING / PROGRESS
#######################
# This script runs both interactively (TTY, where tqdm's in-place bar is
# the right UX) and inside GitHub Actions (no TTY, where tqdm collapses
# to a single line per minute and gives the operator zero signal of
# liveness or failure point). Detect the environment up front and switch
# the progress UX accordingly: tqdm for TTY, periodic timestamped log
# lines + GHA log-grouping commands for CI.
_GHA = os.environ.get("GITHUB_ACTIONS") == "true"
_IS_TTY = sys.stderr.isatty()
_USE_BAR = _IS_TTY and not _GHA


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _gha_group(name: str) -> None:
    if _GHA:
        print(f"::group::{name}", flush=True)


def _gha_endgroup() -> None:
    if _GHA:
        print("::endgroup::", flush=True)


def _gha_warning(msg: str) -> None:
    if _GHA:
        print(f"::warning::{msg}", flush=True)
    else:
        _log(f"WARN: {msg}")


def _gha_error(msg: str) -> None:
    if _GHA:
        print(f"::error::{msg}", flush=True)
    else:
        _log(f"ERROR: {msg}")


class ProgressReporter:
    """tqdm-in-TTY, periodic-log-in-CI progress reporter.

    Drop-in around long loops where stdout is the only signal of
    liveness (GitHub Actions, Docker container logs, etc.). In CI mode
    emits a structured progress line every ``every`` updates plus a
    final summary; in TTY mode delegates to tqdm so the local UX is
    unchanged.
    """

    def __init__(self, total: int, label: str, every: int = 25) -> None:
        self.total: int = total
        self.label: str = label
        self.every: int = max(1, every)
        self.done: int = 0
        self.ok: int = 0
        self.fail: int = 0
        self.start: float = time.monotonic()
        self.bar: Optional[Any] = None
        if _USE_BAR:
            self.bar = tqdm(total=total, desc=label, file=sys.stderr)
        else:
            _log(f"{label}: starting (total={total})")

    def update(self, *, ok: bool = True) -> None:
        self.done += 1
        if ok:
            self.ok += 1
        else:
            self.fail += 1
        if self.bar is not None:
            self.bar.update(1)
            return
        # CI path: log every `every` items + the final tick so the run
        # has visible heartbeat without flooding the log.
        if self.done % self.every == 0 or self.done == self.total:
            elapsed = time.monotonic() - self.start
            rate = self.done / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else 0.0
            _log(
                f"{self.label}: [{self.done}/{self.total}] "
                f"ok={self.ok} fail={self.fail} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={eta:.1f}s"
            )

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            return
        elapsed = time.monotonic() - self.start
        _log(
            f"{self.label}: done [{self.done}/{self.total}] "
            f"ok={self.ok} fail={self.fail} elapsed={elapsed:.1f}s"
        )


#######################
# CODE
#######################

def calculate_rsi(prices: pd.Series, periods: int = 30) -> float:
    """Calculate the Relative Strength Index (RSI) for a given price series.

    Returns NaN if the series is too short to compute the rolling RSI."""
    # Calculate price differences
    delta = prices.diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Guard against the degenerate case where arithmetic collapses to a
    # scalar (e.g. an empty input series). pd.Series arithmetic returns a
    # Series in the common case, but the type system cannot prove that.
    if not isinstance(rsi, pd.Series):
        return float("nan")
    if rsi.empty:
        return float("nan")
    last = rsi.iloc[-1]
    if last is None or pd.isna(last):
        return float("nan")
    return float(last)

def calculate_return_sd(prices: pd.Series, periods: int = 252) -> float:
    """Calculate the standard deviation of returns over the specified period."""
    # Calculate daily returns
    returns = prices.pct_change()

    # Calculate the standard deviation of returns
    return_sd = returns.std()

    # `Series.std()` returns a scalar (or NaN). `pd.isna(scalar)` returns
    # bool, but its stub is widened to bool|NDArray|NDFrame; coerce to bool
    # so the conditional is well-typed.
    if return_sd is None or bool(pd.isna(return_sd)):
        return float("nan")
    return float(return_sd)

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate the maximum drawdown over the given price series."""
    # Calculate cumulative peak
    rolling_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - rolling_max) / rolling_max
    
    # Get the maximum drawdown
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)  # Return as a positive number

async def get_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Get historical price data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

# Semaphore for limiting concurrent requests
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def get_metric_async(session: aiohttp.ClientSession, ticker: str, metric_name: str) -> float:
    """Get a specific metric for a ticker asynchronously."""
    async with SEMAPHORE:  # Limit concurrent requests
        try:
            await asyncio.sleep(REQUEST_DELAY)  # Rate limiting
            stock = yf.Ticker(ticker)

            # Handle custom metrics that need calculation
            if metric_name in ("rsi", "returnSD", "maxDrawdown"):
                hist = await get_historical_data(ticker)
                if hist.empty:
                    return float("nan")
                close = hist['Close']
                if not isinstance(close, pd.Series):
                    return float("nan")
                if metric_name == "rsi":
                    return calculate_rsi(close)
                elif metric_name == "returnSD":
                    return calculate_return_sd(close)
                elif metric_name == "maxDrawdown":
                    return calculate_max_drawdown(close)

            # logMarketCap is derived from the raw marketCap field rather
            # than fetched directly. ln(0) and ln(negative) are undefined;
            # guard them explicitly and return NaN so the mean-imputation
            # path in process_data handles the missing row consistently.
            if metric_name == "logMarketCap":
                raw_market_cap = stock.info.get("marketCap")
                if raw_market_cap is None:
                    return float("nan")
                try:
                    value = float(raw_market_cap)
                except (TypeError, ValueError):
                    return float("nan")
                if value <= 0 or not np.isfinite(value):
                    return float("nan")
                return float(np.log(value))

            # Handle standard yfinance metrics. yfinance occasionally returns
            # non-numeric values (e.g. None, strings) so coerce defensively to
            # honor the declared float contract.
            value = stock.info.get(metric_name)
            if value is None:
                return float("nan")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        except Exception as e:
            print(f"Error fetching {metric_name} for {ticker}: {e}")
            return float("nan")

async def get_company_metrics_async(session: aiohttp.ClientSession, company: str,
                                 all_metrics: Dict[str, str]) -> Dict[str, Any]:
    """Get all metrics for a single company asynchronously."""
    company_data: Dict[str, Any] = {"Ticker": company}

    # Create tasks for all metrics
    tasks = [
        get_metric_async(session, company, yf_metric)
        for metric_name, yf_metric in all_metrics.items()
    ]

    # Wait for all metrics to be fetched
    results = await asyncio.gather(*tasks)

    # Add results to company data
    for (metric_name, _), value in zip(all_metrics.items(), results):
        company_data[metric_name] = value

    return company_data

async def process_companies_async(
    companies: List[str],
    all_metrics: Dict[str, str],
    sector: str = "?",
) -> List[Dict[str, Any]]:
    """Process a batch of companies asynchronously.

    Emits per-batch progress to stdout (heartbeat every 25 completions in CI,
    a tqdm bar locally) plus a one-line per-failure summary so a CI operator
    can identify which ticker is responsible without re-running with -v.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_company_metrics_async(session, company, all_metrics)
            for company in companies
        ]
        progress = ProgressReporter(
            total=len(tasks),
            label=f"[{sector}] companies",
            every=25,
        )
        company_data_list: List[Dict[str, Any]] = []
        try:
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                except Exception as e:
                    progress.update(ok=False)
                    print(f"  [{sector}] company task raised: {e}", flush=True)
                    continue
                company_data_list.append(result)
                # Per-company quality signal: count NaN metrics. A company with
                # all-NaN metrics is yfinance returning nothing useful (delisted,
                # rate-limited, ticker remap, etc.); surface those in the log so
                # a partial run is diagnosable from the CI output alone.
                ticker = result.get("Ticker", "?")
                metric_values = [v for k, v in result.items() if k != "Ticker"]
                nan_count = sum(
                    1 for v in metric_values
                    if v is None or (isinstance(v, float) and pd.isna(v))
                )
                metric_count = len(metric_values)
                all_nan = metric_count > 0 and nan_count == metric_count
                progress.update(ok=not all_nan)
                if all_nan and not _USE_BAR:
                    _log(
                        f"  {sector}/{ticker}: all-nan "
                        f"({nan_count}/{metric_count} metrics missing)"
                    )
        finally:
            progress.close()

    return company_data_list

# Russell 1000 constituents + GICS sector slugs are loaded once at import
# time from the committed CSV at data/russell1000.csv. The path is relative
# to the repository root because the GHA refresh and the local dashboard
# both run from that working directory.
_RUSSELL_1000_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "russell1000.csv",
)
_russell_1000_by_sector: Optional[Dict[str, List[str]]] = None


def _load_russell_1000() -> Dict[str, List[str]]:
    """Read data/russell1000.csv once and group its tickers by sector slug."""
    global _russell_1000_by_sector
    if _russell_1000_by_sector is not None:
        return _russell_1000_by_sector
    df = pd.read_csv(_RUSSELL_1000_PATH)
    grouped: Dict[str, List[str]] = {}
    for ticker, sector in zip(df["Ticker"].astype(str), df["Sector"].astype(str)):
        grouped.setdefault(sector, []).append(ticker)
    _russell_1000_by_sector = grouped
    return _russell_1000_by_sector


async def get_sector_companies(sector: str) -> List[str]:
    """Return Russell 1000 tickers for the requested GICS sector slug."""
    try:
        by_sector = _load_russell_1000()
        if sector in by_sector:
            return list(by_sector[sector])
        print(f"No companies found for sector {sector}")
        return []
    except Exception as e:
        print(f"Error loading sector companies for {sector}: {e}")
        return []

async def process_sector_async(
    sector: str, metrics: Dict[str, Dict[str, str]]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process a single sector asynchronously.

    Returns a (full_df, simple_df) tuple. Either or both may be ``None``
    when the sector had no companies, no data, or raised during processing.
    """
    try:
        companies = await get_sector_companies(sector)

        if not companies:
            print(f"No companies found for sector {sector}")
            return None, None

        # Get raw data
        company_data_list = await process_companies_async(
            companies, metrics["all_metrics"], sector=sector,
        )

        # Convert to DataFrame
        df = pd.DataFrame(company_data_list)

        if df.empty:
            print(f"No data available for sector {sector}")
            return None, None

        # Add sector column before processing
        df["Sector"] = sector

        # Process data
        full_df, simple_df = await process_data(df, metrics)
        return full_df, simple_df
    except Exception as e:
        print(f"Error processing sector {sector}: {e}")
        return None, None

async def process_data(
    df: pd.DataFrame, metrics: Dict[str, Dict[str, str]]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process the collected data by calculating z-scores and composite scores."""
    if df.empty:
        print("No data was collected successfully.")
        return None, None

    # yfinance occasionally returns +/- inf for ratios with near-zero
    # denominators (e.g. trailingPE for a company with EPS just barely
    # positive). pd.dropna does NOT treat inf as missing, so a single
    # inf value silently poisons scipy.stats.zscore (mean -> inf, std ->
    # inf, every z -> NaN), which then wipes the entire sector through
    # the 2.5-sigma outlier filter below. Coerce inf to NaN up front so
    # the existing missing-data paths (dropna for PE, mean-imputation
    # for the X1/X2/X3 metrics) handle these rows correctly.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Calculate z-scores for each metric group
    for metric_group in [
        "x1_risk_metrics",
        "x2_momentum_metrics",
        "x3_quality_metrics",
        "x5_size_metrics",
        "x6_growth_metrics",
    ]:
        for metric_name in metrics[metric_group].keys():
            if metric_name in df.columns:
                df.loc[:, f"{metric_name}_ZScore"] = zscore(df[metric_name].fillna(df[metric_name].mean()),
                                                   nan_policy='omit')

    # Calculate PE Z-score separately (our target variable)
    if "PE" in df.columns:
        # Remove companies with no P/E values
        df = df.dropna(subset=["PE"]).copy()  # Create explicit copy
        # Calculate z-score only for remaining companies
        df.loc[:, "PE_ZScore"] = zscore(df["PE"], nan_policy='omit')

    # Calculate composite scores for each category
    df.loc[:, "Risk_Score"] = df[[f"{metric}_ZScore" for metric in X1_RISK_METRICS.keys()
                          if f"{metric}_ZScore" in df.columns]].mean(axis=1)
    df.loc[:, "Momentum_Score"] = df[[f"{metric}_ZScore" for metric in X2_MOMENTUM_METRICS.keys()
                            if f"{metric}_ZScore" in df.columns]].mean(axis=1)
    df.loc[:, "Quality_Score"] = df[[f"{metric}_ZScore" for metric in X3_QUALITY_METRICS.keys()
                             if f"{metric}_ZScore" in df.columns]].mean(axis=1)
    df.loc[:, "Size_Score"] = df[[f"{metric}_ZScore" for metric in X5_SIZE_METRICS.keys()
                          if f"{metric}_ZScore" in df.columns]].mean(axis=1)
    df.loc[:, "Growth_Score"] = df[[f"{metric}_ZScore" for metric in X6_GROWTH_METRICS.keys()
                            if f"{metric}_ZScore" in df.columns]].mean(axis=1)

    # Filter out data points with extreme z-scores. The threshold is
    # applied to every composite so that one runaway factor cannot
    # silently rescue an obviously-broken row. 3.0-sigma rather than
    # the historical 2.5-sigma: at 5 predictors, six independent AND'd
    # filters at 2.5-sigma drop too much of the smallest sectors
    # (energy at n ~31 going to ~25 leaves n/p = 5, which is below
    # the rule-of-thumb >=10 needed for a stable per-sector regression).
    # 3.0-sigma keeps the universe at ~98% of post-PE-dropna rows.
    mask = (abs(df["Risk_Score"]) <= 3.0) & \
           (abs(df["Momentum_Score"]) <= 3.0) & \
           (abs(df["Quality_Score"]) <= 3.0) & \
           (abs(df["Size_Score"]) <= 3.0) & \
           (abs(df["Growth_Score"]) <= 3.0) & \
           (abs(df["PE_ZScore"]) <= 3.0)
    df_filtered = df[mask]
    # df[boolean_mask] returns a DataFrame in this context; narrow for pyright.
    assert isinstance(df_filtered, pd.DataFrame)
    df = df_filtered

    # Keep only essential columns
    columns_to_keep = [
        "Sector", "Ticker",
        "Risk_Score", "Momentum_Score", "Quality_Score", "Size_Score", "Growth_Score",
    ]
    if "PE" in df.columns:
        columns_to_keep.extend(["PE", "PE_ZScore"])

    simple_df_raw = df[columns_to_keep]
    # df[list_of_str_columns] semantically always returns a DataFrame, but
    # pyright's pandas stubs widen the result type. Narrow explicitly.
    assert isinstance(simple_df_raw, pd.DataFrame)
    simple_df = simple_df_raw.copy()

    return df, simple_df

async def analyze_sectors_async(sectors: List[str] = SECTORS) -> Optional[pd.DataFrame]:
    """Analyze multiple sectors with controlled concurrency.

    Returns the combined simplified per-ticker DataFrame, or ``None`` if no
    sector produced any data. Exits with status 1 if any sector silently
    yielded zero rows so the caller (typically a GHA workflow) can fail
    visibly instead of committing a partial dataset.
    """
    # Process the largest sectors first. The yfinance rate-limit budget is
    # freshest at the start of a run; if Yahoo throttles partway through,
    # we'd rather lose a small sector than a large one. The previous
    # alphabetical order had technology near the end and it was silently
    # wiped out by rate-limiting in production.
    overall_start = time.monotonic()
    _log(f"=== Sector refresh starting: {len(sectors)} sectors ===")
    _log(f"runtime: GITHUB_ACTIONS={_GHA} TTY={_IS_TTY}")

    candidate_counts: Dict[str, int] = {}
    for sector in sectors:
        companies = await get_sector_companies(sector)
        candidate_counts[sector] = len(companies)
    ordered_sectors = sorted(sectors, key=lambda s: candidate_counts.get(s, 0), reverse=True)

    _log("Processing order (largest sector first; preserves rate-limit budget for biggest):")
    for i, s in enumerate(ordered_sectors, 1):
        _log(f"  {i:2d}/{len(ordered_sectors)} {s} ({candidate_counts[s]} companies)")

    all_data: List[pd.DataFrame] = []
    all_data_full: List[pd.DataFrame] = []
    failed_sectors: List[Tuple[str, int]] = []

    for sector_idx, sector in enumerate(ordered_sectors, 1):
        await asyncio.sleep(INDUSTRY_DELAY)  # Rate limiting between sectors
        sector_start = time.monotonic()
        header = (
            f"Sector {sector_idx}/{len(ordered_sectors)}: {sector} "
            f"({candidate_counts[sector]} companies)"
        )
        _gha_group(header)
        _log(f">>> {header}")
        full_data, simple_data = await process_sector_async(sector, METRICS)
        sector_elapsed = time.monotonic() - sector_start
        if simple_data is None or simple_data.empty:
            failed_sectors.append((sector, candidate_counts[sector]))
            _log(
                f"<<< {sector}: 0 surviving rows from {candidate_counts[sector]} "
                f"candidates, elapsed={sector_elapsed:.1f}s"
            )
            _gha_warning(
                f"{sector}: 0 surviving rows from {candidate_counts[sector]} candidates"
            )
        else:
            n_rows = len(simple_data)
            n_full = len(full_data) if full_data is not None else 0
            _log(
                f"<<< {sector}: {n_rows} rows after filter "
                f"(full={n_full}, from {candidate_counts[sector]} candidates) "
                f"elapsed={sector_elapsed:.1f}s"
            )
            all_data.append(simple_data)
            if full_data is not None:
                all_data_full.append(full_data)
        _gha_endgroup()

    overall_elapsed = time.monotonic() - overall_start

    if failed_sectors:
        _gha_error(f"{len(failed_sectors)} sector(s) yielded zero surviving rows")
        _log("SECTOR FAILURES (yielded zero surviving rows):")
        for sector, n_candidates in failed_sectors:
            _log(f"  {sector}: 0 rows from {n_candidates} candidates")

    if not all_data:
        _log("No data was collected successfully.")
        sys.exit(1)

    # Combine all sector data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data_full = pd.concat(all_data_full, ignore_index=True)

    # Save to CSV
    combined_data.to_csv("sector_analysis.csv", index=False)
    combined_data_full.to_csv("sector_analysis_full.csv", index=False)

    _log(
        f"=== Refresh complete: {len(combined_data)} rows across {len(all_data)} "
        f"sectors, total elapsed {overall_elapsed:.1f}s ==="
    )

    # Partial-success: write what we got so a follow-up can patch the gap,
    # but exit non-zero so CI / the operator notices instead of pushing
    # silently-broken data.
    if failed_sectors:
        sys.exit(1)

    return combined_data

if __name__ == "__main__":
    # Run the async analysis
    results = asyncio.run(analyze_sectors_async())
