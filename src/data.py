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
X3_QUALITY_METRICS = {
    "ROE": "returnOnEquity",                # Return on Equity
    "ROA": "returnOnAssets",                # Return on Assets
    "OperatingMargin": "operatingMargins"   # Operating efficiency
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
                    rsi = calculate_rsi(close)
                    print(f"{metric_name}: {rsi}")
                    return rsi
                elif metric_name == "returnSD":
                    sd = calculate_return_sd(close)
                    print(f"{metric_name}: {sd}")
                    return sd
                elif metric_name == "maxDrawdown":
                    max_drawdown = calculate_max_drawdown(close)
                    print(f"{metric_name}: {max_drawdown}")
                    return max_drawdown

            # logMarketCap is derived from the raw marketCap field rather
            # than fetched directly. ln(0) and ln(negative) are undefined;
            # guard them explicitly and return NaN so the mean-imputation
            # path in process_data handles the missing row consistently.
            if metric_name == "logMarketCap":
                raw_market_cap = stock.info.get("marketCap")
                print(f"{metric_name}: raw marketCap={raw_market_cap}")
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
            print(f"{metric_name}: {value}")
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

async def process_companies_async(companies: List[str], all_metrics: Dict[str, str]) -> List[Dict[str, Any]]:
    """Process a batch of companies asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for company in companies:
            task = get_company_metrics_async(session, company, all_metrics)
            tasks.append(task)
        
        # Use tqdm to show progress
        company_data_list = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing companies"):
            try:
                result = await future
                company_data_list.append(result)
            except Exception as e:
                print(f"Error processing company: {e}")
    
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
        company_data_list = await process_companies_async(companies, metrics["all_metrics"])

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

    # Filter out data points with extreme z-scores. The threshold is
    # applied to every composite so that one runaway factor cannot
    # silently rescue an obviously-broken row.
    mask = (abs(df["Risk_Score"]) <= 2.5) & \
           (abs(df["Momentum_Score"]) <= 2.5) & \
           (abs(df["Quality_Score"]) <= 2.5) & \
           (abs(df["Size_Score"]) <= 2.5) & \
           (abs(df["PE_ZScore"]) <= 2.5)
    df_filtered = df[mask]
    # df[boolean_mask] returns a DataFrame in this context; narrow for pyright.
    assert isinstance(df_filtered, pd.DataFrame)
    df = df_filtered

    # Keep only essential columns
    columns_to_keep = ["Sector", "Ticker", "Risk_Score", "Momentum_Score", "Quality_Score", "Size_Score"]
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
    candidate_counts: Dict[str, int] = {}
    for sector in sectors:
        companies = await get_sector_companies(sector)
        candidate_counts[sector] = len(companies)
    ordered_sectors = sorted(sectors, key=lambda s: candidate_counts.get(s, 0), reverse=True)

    all_data: List[pd.DataFrame] = []
    all_data_full: List[pd.DataFrame] = []
    failed_sectors: List[Tuple[str, int]] = []

    for sector in tqdm(ordered_sectors, desc="Processing sectors"):
        await asyncio.sleep(INDUSTRY_DELAY)  # Rate limiting between sectors
        full_data, simple_data = await process_sector_async(sector, METRICS)
        if simple_data is None or simple_data.empty:
            failed_sectors.append((sector, candidate_counts[sector]))
            continue
        all_data.append(simple_data)
        if full_data is not None:
            all_data_full.append(full_data)

    if failed_sectors:
        print("\n!!! SECTOR FAILURES — yielded zero surviving rows !!!")
        for sector, n_candidates in failed_sectors:
            print(f"  {sector}: 0 rows from {n_candidates} candidates")

    if not all_data:
        print("No data was collected successfully.")
        sys.exit(1)

    # Combine all sector data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data_full = pd.concat(all_data_full, ignore_index=True)

    # Save to CSV
    combined_data.to_csv("sector_analysis.csv", index=False)
    combined_data_full.to_csv("sector_analysis_full.csv", index=False)

    # Partial-success: write what we got so a follow-up can patch the gap,
    # but exit non-zero so CI / the operator notices instead of pushing
    # silently-broken data.
    if failed_sectors:
        sys.exit(1)

    return combined_data

if __name__ == "__main__":
    # Run the async analysis
    results = asyncio.run(analyze_sectors_async())
