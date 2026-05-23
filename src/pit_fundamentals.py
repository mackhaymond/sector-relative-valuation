"""Point-in-time fundamentals from SimFin's bulk free-tier datasets.

Fixes the look-ahead bias that Phase 3's backtest (`src/backtest.py`)
documented as its dominant caveat: instead of reusing today's
trailing-PE and today's composite z-score at every historical snapshot,
this module fetches the most-recent SimFin filing whose `Publish Date`
is on-or-before the snapshot's `as_of` date and derives the per-snapshot
metric set from that filing.

Design notes:

* Bulk-download pattern. Per-ticker REST calls would burn through free-
  tier credits in minutes and rate-limit unpredictably. We download the
  full US income/balance TTM and annual datasets once (cached to
  `backtest_cache/simfin/` for `refresh_days=30`) and do all filtering
  in-process.
* PUBLISH_DATE, not REPORT_DATE. A 2023-Q4 filing has a fiscal period
  ending 2023-12-31 but doesn't become public until February 2024. Using
  REPORT_DATE as the as-of cutoff would leak two months of look-ahead;
  PUBLISH_DATE is what an investor on the snapshot date could actually
  see.
* Derived ratios (ROE / ROA / margins) are computed locally from raw
  income + balance lines. SimFin's `derived` bulk dataset returns HTTP
  500 on the free tier at the moment, so we sidestep the dependency
  entirely and avoid the extra download.
* Module-level state. The SimFin frames are loaded once on first use
  and held in process memory; per-snapshot calls only do an in-memory
  filter. This is the standard SimFin SDK pattern and is necessary for
  per-snapshot per-ticker fetches to run in reasonable time across a
  36-month x 800-ticker backtest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIMFIN_CACHE_DIR = ROOT / "backtest_cache" / "simfin"

# Output dict keys returned by `fetch_pit_metrics`. These match the
# metric names in src/data.py's X1/X2/X3/X5/X6/Y dicts (less the price-
# derived metrics, which stay in src/backtest.py via the yfinance price
# cache). Listed here as a single source of truth so callers / tests
# don't drift from the contract.
PIT_METRIC_KEYS: Tuple[str, ...] = (
    "DebtToEquity",
    "EarningsGrowth",
    "ROE",
    "ROA",
    "OperatingMargin",
    "EBITDAMargin",
    "RevenueGrowth",
    "TTM_EPS",
    "SharesOutstanding",
    "PublishDate",
)


# Module-level cache. None until first fetch_pit_metrics call.
_income_ttm: Optional[pd.DataFrame] = None
_balance_ttm: Optional[pd.DataFrame] = None


def _load_api_key() -> str:
    """Read SIMFIN_API_KEY from env, falling back to a .env file.

    Never logs the key. Raises with a non-leaking message if missing.
    """
    key = os.environ.get("SIMFIN_API_KEY")
    if key:
        return key
    # Fallback for local dev where the user sources .env via direnv /
    # python-dotenv but the env var isn't otherwise set.
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("SIMFIN_API_KEY")
    if not key:
        raise RuntimeError(
            "SIMFIN_API_KEY not set. Add it to .env (see .env.example) or "
            "export it in the shell before running the backtest."
        )
    return key


def _ensure_datasets_loaded() -> None:
    """Download (or read from cache) the SimFin TTM income + balance.

    Idempotent: subsequent calls return immediately. The SimFin SDK
    handles on-disk caching via ``refresh_days``; we set 30 days for the
    free tier (datasets update infrequently).
    """
    global _income_ttm, _balance_ttm
    if _income_ttm is not None and _balance_ttm is not None:
        return

    import simfin as sf
    from simfin.names import PUBLISH_DATE, REPORT_DATE, TICKER

    SIMFIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sf.set_api_key(_load_api_key())
    sf.set_data_dir(str(SIMFIN_CACHE_DIR))

    parse_dates = [REPORT_DATE, PUBLISH_DATE, "Restated Date"]
    index = [TICKER, REPORT_DATE]

    _income_ttm = sf.load_income(
        variant="ttm",
        market="us",
        parse_dates=parse_dates,
        index=index,
        refresh_days=30,
    )
    _balance_ttm = sf.load_balance(
        variant="ttm",
        market="us",
        parse_dates=parse_dates,
        index=index,
        refresh_days=30,
    )


def _filings_as_of(
    df: pd.DataFrame, ticker: str, as_of: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """Sub-frame of ``df`` for ``ticker`` with PUBLISH_DATE <= ``as_of``.

    Returns None if the ticker is absent or has no qualifying filings.
    The returned frame is sorted by Publish Date ascending; callers
    typically want ``.iloc[-1]`` (latest) and ``.iloc[-5]`` (year-ago
    for TTM growth deltas).
    """
    if ticker not in df.index.get_level_values(0):
        return None
    sub = df.loc[ticker]
    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T
    mask = sub["Publish Date"] <= as_of
    qualifying = sub[mask]
    assert isinstance(qualifying, pd.DataFrame)
    if qualifying.empty:
        return None
    sorted_qualifying = qualifying.sort_values(by="Publish Date")
    assert isinstance(sorted_qualifying, pd.DataFrame)
    return sorted_qualifying


def _safe_div(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    """num / denom guarding against None, NaN, and zero denominator."""
    if num is None or denom is None:
        return None
    if not np.isfinite(num) or not np.isfinite(denom) or denom == 0:
        return None
    return float(num) / float(denom)


def _ttm_growth(latest: Optional[float], year_ago: Optional[float]) -> Optional[float]:
    """Year-over-year growth ratio matching yfinance's revenueGrowth /
    earningsGrowth definition: (latest - year_ago) / |year_ago|.

    Uses absolute value of the denominator so a swing from loss to gain
    doesn't flip sign. Returns None if either side is missing or the
    year-ago value is zero.
    """
    if latest is None or year_ago is None:
        return None
    if not np.isfinite(latest) or not np.isfinite(year_ago) or year_ago == 0:
        return None
    return (float(latest) - float(year_ago)) / abs(float(year_ago))


def fetch_pit_metrics(
    ticker: str, as_of: pd.Timestamp
) -> Dict[str, Optional[float]]:
    """Return PIT-correct fundamental metrics for ``ticker`` at ``as_of``.

    Uses the most-recent SimFin filing with Publish Date on-or-before
    ``as_of`` (no look-ahead). Returns a dict with every key in
    ``PIT_METRIC_KEYS``; missing values are None — the caller MUST
    handle Nones rather than imputing silently. Price-derived metrics
    (MaxDrawdown, ReturnSD, RSI, PriceChange12M, MarketCap) are NOT in
    the dict; they're computed by the caller from the yfinance price
    cache because price history is the one PIT-correct input yfinance
    free tier exposes.
    """
    _ensure_datasets_loaded()
    assert _income_ttm is not None and _balance_ttm is not None

    out: Dict[str, Optional[float]] = {k: None for k in PIT_METRIC_KEYS}

    inc = _filings_as_of(_income_ttm, ticker, as_of)
    bal = _filings_as_of(_balance_ttm, ticker, as_of)
    if inc is None and bal is None:
        return out

    # Latest filing values.
    latest_inc = inc.iloc[-1] if inc is not None and not inc.empty else None
    latest_bal = bal.iloc[-1] if bal is not None and not bal.empty else None

    if latest_inc is not None:
        out["PublishDate"] = latest_inc["Publish Date"]
    elif latest_bal is not None:
        out["PublishDate"] = latest_bal["Publish Date"]

    revenue = float(latest_inc["Revenue"]) if latest_inc is not None and pd.notna(latest_inc.get("Revenue")) else None
    net_income = float(latest_inc["Net Income"]) if latest_inc is not None and pd.notna(latest_inc.get("Net Income")) else None
    op_income = (
        float(latest_inc["Operating Income (Loss)"])
        if latest_inc is not None and pd.notna(latest_inc.get("Operating Income (Loss)"))
        else None
    )
    dep_amort = (
        float(latest_inc["Depreciation & Amortization"])
        if latest_inc is not None and pd.notna(latest_inc.get("Depreciation & Amortization"))
        else None
    )

    total_equity = (
        float(latest_bal["Total Equity"])
        if latest_bal is not None and pd.notna(latest_bal.get("Total Equity"))
        else None
    )
    total_assets = (
        float(latest_bal["Total Assets"])
        if latest_bal is not None and pd.notna(latest_bal.get("Total Assets"))
        else None
    )
    long_term_debt = (
        float(latest_bal["Long Term Debt"])
        if latest_bal is not None and pd.notna(latest_bal.get("Long Term Debt"))
        else None
    )
    short_term_debt = (
        float(latest_bal["Short Term Debt"])
        if latest_bal is not None and pd.notna(latest_bal.get("Short Term Debt"))
        else None
    )
    shares_basic = (
        float(latest_bal["Shares (Basic)"])
        if latest_bal is not None and pd.notna(latest_bal.get("Shares (Basic)"))
        else (
            float(latest_inc["Shares (Basic)"])
            if latest_inc is not None and pd.notna(latest_inc.get("Shares (Basic)"))
            else None
        )
    )

    # Derived ratios. Match the SimFin/yfinance unit conventions where
    # possible: margins and returns are fractions (0.15 = 15%), not
    # percentages.
    out["ROE"] = _safe_div(net_income, total_equity)
    out["ROA"] = _safe_div(net_income, total_assets)
    out["OperatingMargin"] = _safe_div(op_income, revenue)
    # EBITDA = Operating Income + D&A. SimFin's TTM income statement
    # does not carry a direct EBITDA line; the OpInc+D&A reconstruction
    # is the conventional approximation and matches the value the
    # yfinance `ebitdaMargins` field reports for non-financial sectors.
    ebitda = (op_income + dep_amort) if (op_income is not None and dep_amort is not None) else None
    out["EBITDAMargin"] = _safe_div(ebitda, revenue)
    total_debt = (long_term_debt or 0.0) + (short_term_debt or 0.0)
    # If both debt lines are missing, treat as None rather than 0 to
    # avoid claiming a debt-free balance sheet for tickers SimFin just
    # doesn't carry.
    if long_term_debt is None and short_term_debt is None:
        total_debt = None  # type: ignore[assignment]
    # yfinance reports debtToEquity in percent (e.g. 143 means 143%).
    # Match that convention so the downstream z-scoring sees the same
    # scale it saw under the Phase 3 (yfinance) path.
    de = _safe_div(total_debt, total_equity)
    out["DebtToEquity"] = (de * 100.0) if de is not None else None
    out["TTM_EPS"] = _safe_div(net_income, shares_basic)
    out["SharesOutstanding"] = shares_basic

    # TTM growth deltas. We need the filing whose Publish Date is closest
    # to (as_of - 12 months) for the year-ago snapshot. Easiest: take
    # the row of `inc` whose Publish Date is the largest <= as_of - 1yr.
    year_ago_cutoff = as_of - pd.DateOffset(years=1)
    year_ago_inc = (
        inc[inc["Publish Date"] <= year_ago_cutoff].iloc[-1]
        if inc is not None and (inc["Publish Date"] <= year_ago_cutoff).any()
        else None
    )
    if year_ago_inc is not None:
        ya_rev = float(year_ago_inc["Revenue"]) if pd.notna(year_ago_inc.get("Revenue")) else None
        ya_ni = float(year_ago_inc["Net Income"]) if pd.notna(year_ago_inc.get("Net Income")) else None
        ya_shares = (
            float(year_ago_inc["Shares (Basic)"])
            if pd.notna(year_ago_inc.get("Shares (Basic)"))
            else None
        )
        out["RevenueGrowth"] = _ttm_growth(revenue, ya_rev)
        ya_eps = _safe_div(ya_ni, ya_shares)
        out["EarningsGrowth"] = _ttm_growth(out["TTM_EPS"], ya_eps)

    return out
