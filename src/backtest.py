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

import pandas as pd
import yfinance as yf

# Repository root: this file lives at <repo>/src/backtest.py.
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "backtest_cache"
PRICE_CACHE_PATH = CACHE_DIR / "prices.pkl"

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
