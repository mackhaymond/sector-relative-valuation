"""Sector-relative valuation backtest.

Monthly-rebalanced long-short backtest of the deviation signal: at each
snapshot date, within each sector, rank tickers by ``actual_PE_t -
predicted_PE_t`` (where predicted_PE_t is the within-sector OLS fit of
PE_t on composite_z_score, refit per snapshot), long the cheapest
quintile, short the richest, equal-weight within each leg, then
aggregate across sectors with equal sector weights.

Methodological caveats (see BACKTEST.md for the full discussion):

* Fundamentals (the per-ticker composite_z_score and the per-sector
  Ridge weights baked into it) are CURRENT, not point-in-time. Using
  today's factor values to backtest yesterday's signal is a look-ahead
  bias of unknown magnitude. yfinance free tier does not expose
  historical fundamentals.
* Per-ticker TTM EPS is also current. We compute it as
  ``last_price_i / current_PE_i`` and reuse the same scalar at every
  snapshot, so actual_PE_t moves only with price; the EPS denominator
  is constant per ticker over the backtest window.
* Survivorship bias: the universe at every snapshot is today's Russell
  1000 (as captured in sector_analysis.csv). Companies that were in the
  index historically but have since been delisted, merged, or demoted
  are absent.
* The only PIT-correct input is the price series. yfinance.history /
  yf.download with ``auto_adjust=True`` returns split- and
  dividend-adjusted prices on the actual historical calendar date.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Repository root: this file lives at <repo>/src/backtest.py.
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "backtest_cache"
PRICE_CACHE_PATH = CACHE_DIR / "prices.pkl"
SECTOR_ANALYSIS_PATH = ROOT / "sector_analysis.csv"

# Minimum tickers needed per sector at a snapshot to fit the within-
# sector OLS and form 5 quintiles cleanly. Below this we skip the
# sector for that snapshot rather than produce a noisy single-bucket
# "portfolio". 15 = at least ~3 names per quintile on average.
_MIN_SECTOR_N = 15

# Forward-return horizon in months. Locked at 1mo per spec; not exposed
# to the CLI to prevent the obvious p-hacking path ("try 3mo, then
# 6mo, until something looks good"). Documented in BACKTEST.md.
_FORWARD_MONTHS = 1

# yf.download chokes when the ticker list is enormous (rate-limit
# rejections become probabilistic well before the official cap).
# 200 tickers per batch is a comfortable middle ground: large enough
# that 795 Russell 1000 names fan out to four calls, small enough that
# a single throttled batch doesn't lose the whole pull.
_DOWNLOAD_CHUNK_SIZE = 200


def _download_close_chunk(
    tickers: List[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Fetch adjusted close prices for one ticker batch via yfinance.

    Returns a DataFrame indexed by trading date with one column per
    ticker. Tickers yfinance can't resolve are silently absent from the
    returned frame (yfinance drops them); the caller should not assume
    every requested ticker is present.
    """
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()
    # yfinance returns different shapes for a single-ticker vs
    # multi-ticker request. Normalize to "rows = date, columns = ticker"
    # of adjusted close prices.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        close = data["Close"]
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        close = data[["Close"]].copy()
        close.columns = pd.Index([tickers[0]])
    if not isinstance(close, pd.DataFrame):
        # `data['Close']` on a single-level frame returns a Series; coerce.
        close = close.to_frame()
    return close


def fetch_prices(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return split- and dividend-adjusted close prices for ``tickers``.

    Caches the result to ``cache_path`` (pickle). If the cache exists
    and covers the requested ``[start, end]`` window for every requested
    ticker, the cached frame is returned unchanged; otherwise the full
    range is refetched and the cache overwritten.

    The cache is intentionally coarse: a finer-grained "patch the gap"
    strategy invites silent staleness bugs (e.g. a ticker delisted after
    the cache was built keeps returning the old final price). Refetching
    the whole window is cheap (minutes) and safer.
    """
    if cache_path is None:
        cache_path = PRICE_CACHE_PATH

    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                cached = pickle.load(fh)
            if (
                isinstance(cached, pd.DataFrame)
                and not cached.empty
                and isinstance(cached.index, pd.DatetimeIndex)
            ):
                idx_min = cached.index.min()
                idx_max = cached.index.max()
                # DatetimeIndex.min/.max return Timestamp at runtime but
                # pandas-stubs widens the return to a union; narrow with
                # isinstance so the < and > comparisons type-check.
                if isinstance(idx_min, pd.Timestamp) and isinstance(
                    idx_max, pd.Timestamp
                ):
                    covers_window = idx_min <= start and idx_max >= end
                    covers_tickers = set(tickers).issubset(set(cached.columns))
                    if covers_window and covers_tickers:
                        return cached
        except (pickle.UnpicklingError, EOFError, AttributeError):
            # Corrupt cache - fall through to refetch.
            pass

    parts: List[pd.DataFrame] = []
    for i in range(0, len(tickers), _DOWNLOAD_CHUNK_SIZE):
        chunk = tickers[i : i + _DOWNLOAD_CHUNK_SIZE]
        part = _download_close_chunk(chunk, start, end)
        if not part.empty:
            parts.append(part)

    if not parts:
        raise RuntimeError(
            f"yfinance returned no price data for any of {len(tickers)} tickers "
            f"between {start.date()} and {end.date()}"
        )

    combined = pd.concat(parts, axis=1)
    # Defensive: a ticker can appear in multiple chunks only if the
    # caller passed duplicates. Dedupe by keeping the first.
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(combined, fh)

    return combined


def load_sector_analysis(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the per-ticker sector / composite_z_score / PE snapshot.

    This is the current cross-section produced by `src/data.py` plus
    `src/generate_weights.py`. The columns this backtest needs:
    Ticker, Sector, composite_z_score, PE. composite_z_score and PE are
    "as of now" — see module docstring for the look-ahead implications.
    """
    if path is None:
        path = SECTOR_ANALYSIS_PATH
    df = pd.read_csv(path)
    required = {"Ticker", "Sector", "composite_z_score", "PE"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"sector_analysis.csv is missing required columns: {sorted(missing)}. "
            f"Run `uv run python src/generate_weights.py` first."
        )
    return df.dropna(subset=["composite_z_score", "PE"]).copy()


def build_snapshot_dates(
    price_index: pd.DatetimeIndex, months: int
) -> List[pd.Timestamp]:
    """Return ``months`` monthly trading-day snapshot dates.

    Each snapshot is the first available trading day on/after a calendar
    month-start. The last snapshot is constrained so its t+1mo forward
    date is still inside ``price_index`` (we need both ends to compute a
    forward return). Returns the most recent ``months`` snapshots that
    satisfy that constraint.
    """
    if len(price_index) == 0:
        return []
    last = price_index.max()
    first = price_index.min()
    # Generate calendar month starts and snap each forward to the first
    # actual trading day on/after it (handles weekends + market holidays).
    month_starts = pd.date_range(start=first, end=last, freq="MS")
    snaps: List[pd.Timestamp] = []
    for d in month_starts:
        candidates = price_index[price_index >= d]
        if len(candidates) == 0:
            continue
        first = candidates[0]
        assert isinstance(first, pd.Timestamp)
        snaps.append(first)
    # Drop snapshots without a t+_FORWARD_MONTHS counterpart in-window.
    valid: List[pd.Timestamp] = []
    for s in snaps:
        target = s + pd.DateOffset(months=_FORWARD_MONTHS)
        if (price_index >= target).any():
            valid.append(s)
    return valid[-months:]


def _next_trading_day(
    price_index: pd.DatetimeIndex, target: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """First trading day in ``price_index`` on/after ``target``, or None."""
    candidates = price_index[price_index >= target]
    if len(candidates) == 0:
        return None
    first = candidates[0]
    assert isinstance(first, pd.Timestamp)
    return first


def build_snapshot_panel(
    sa: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Static per-ticker inputs reused at every snapshot.

    Returns a DataFrame indexed by Ticker with columns Sector,
    composite_z_score, ttm_eps. ``ttm_eps`` is the look-ahead-affected
    EPS proxy: today's price / today's PE. Tickers without a usable EPS
    (no current PE, no recent price, non-positive implied EPS) are
    excluded — they cannot produce a PE_t at any snapshot.
    """
    panel_raw = sa.set_index("Ticker")[["Sector", "composite_z_score", "PE"]]
    assert isinstance(panel_raw, pd.DataFrame)
    panel = panel_raw.copy()
    # "Today's" price = most recent value forward-filled per ticker, so
    # a ticker that stopped trading a few days before the cache cutoff
    # still gets its last known price as the EPS reference.
    last_prices = prices.ffill().iloc[-1]
    panel["last_price"] = last_prices.reindex(panel.index)
    panel["ttm_eps"] = panel["last_price"] / panel["PE"]
    panel = panel.dropna(subset=["last_price", "ttm_eps"])
    # Negative implied EPS means yfinance reported a negative trailing
    # P/E (some loss-making names do come through that way). The signed
    # PE_t would still be meaningful, but the regression target is
    # untrustworthy and the dashboard does not surface negative-PE
    # tickers either. Drop them.
    filtered = panel[panel["ttm_eps"] > 0]
    assert isinstance(filtered, pd.DataFrame)
    return filtered


def _historical_pe(
    price_at_t: pd.Series, ttm_eps: pd.Series
) -> pd.Series:
    """PE_t = price_t / current_TTM_EPS, aligned per ticker."""
    common = price_at_t.index.intersection(ttm_eps.index)
    p = price_at_t.reindex(common)
    e = ttm_eps.reindex(common)
    pe = p / e
    return pe.replace([np.inf, -np.inf], np.nan).dropna()


def compute_snapshot_signal(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    forward_date: pd.Timestamp,
) -> pd.DataFrame:
    """Per-ticker deviation and forward return for one snapshot.

    Returns a DataFrame with columns: Ticker, Sector, composite_z_score,
    pe_t, predicted_pe, deviation, fwd_return. Tickers that lack a
    price at t or t+1mo, or whose sector has fewer than _MIN_SECTOR_N
    surviving names at the snapshot, are dropped.
    """
    if snapshot_date not in prices.index or forward_date not in prices.index:
        return pd.DataFrame()
    price_t = prices.loc[snapshot_date].dropna()
    price_next = prices.loc[forward_date].dropna()
    if not isinstance(price_t, pd.Series) or not isinstance(price_next, pd.Series):
        return pd.DataFrame()

    ttm_eps = panel["ttm_eps"]
    assert isinstance(ttm_eps, pd.Series)
    pe_t = _historical_pe(price_t, ttm_eps)
    pe_t.name = "pe_t"
    if pe_t.empty:
        return pd.DataFrame()

    df = panel.join(pe_t, how="inner")
    df["price_t"] = price_t.reindex(df.index)
    df["price_next"] = price_next.reindex(df.index)
    df = df.dropna(subset=["price_t", "price_next"])
    df["fwd_return"] = (df["price_next"] - df["price_t"]) / df["price_t"]
    df = df.dropna(subset=["fwd_return", "pe_t", "composite_z_score"])

    # Within each sector: 1D OLS of pe_t on composite_z_score (matches
    # the dashboard's per-sector scatter fit at src/dashboard.py:500).
    # Skip sectors below the n-floor — quintiles aren't meaningful and
    # the OLS slope is dominated by 2-3 names.
    pieces: List[pd.DataFrame] = []
    for sector, sub in df.groupby("Sector", sort=False):
        if len(sub) < _MIN_SECTOR_N:
            continue
        x = sub["composite_z_score"].to_numpy(dtype=float)
        y = sub["pe_t"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        sub = sub.copy()
        sub["predicted_pe"] = slope * x + intercept
        sub["deviation"] = sub["pe_t"] - sub["predicted_pe"]
        pieces.append(sub)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces)
    out = out.reset_index().rename(columns={"index": "Ticker"})
    out["snapshot"] = snapshot_date
    out["forward"] = forward_date
    cols = [
        "snapshot",
        "forward",
        "Ticker",
        "Sector",
        "composite_z_score",
        "pe_t",
        "predicted_pe",
        "deviation",
        "fwd_return",
    ]
    projected = out[cols]
    assert isinstance(projected, pd.DataFrame)
    return projected
