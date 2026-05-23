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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

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


# Number of quintile portfolios formed within each sector at each
# snapshot. Locked at 5 (the spec); changing this would invalidate the
# IC/Sharpe comparisons to standard quant convention.
_N_QUINTILES = 5

# Months in a year, used to annualize monthly statistics. Kept as a
# constant so the formula reads `mean / std * sqrt(_MONTHS_PER_YEAR)`
# rather than the magic `12` everywhere.
_MONTHS_PER_YEAR = 12


@dataclass
class BacktestMetrics:
    """Top-line statistics from a completed backtest run.

    All return-based fields are net of the round-trip transaction cost
    the backtest was configured with (see ``cost_bps``). The IC fields
    are signal-side (deviation vs forward return) and unaffected by
    transaction costs.
    """

    months: int
    snapshots: int
    cost_bps: float
    mean_ic: float
    ic_t_stat: float
    ic_information_ratio: float
    ls_mean_monthly_return: float
    ls_std_monthly_return: float
    ls_sharpe_annualized: float
    ls_cumulative_return: float
    ls_max_drawdown: float
    ls_hit_rate: float


def _spearman_ic(deviation: pd.Series, fwd_return: pd.Series) -> float:
    """Spearman rank correlation between deviation and forward return.

    Sign convention: negative deviation (cheap) should predict positive
    forward return, so a working signal yields a NEGATIVE Spearman
    coefficient. We do NOT flip the sign here — callers interpret the
    sign themselves so the raw IC stays consistent across reports.

    Returns NaN if either input has fewer than 3 non-null observations
    or is constant (scipy raises in both cases).
    """
    paired = pd.concat([deviation, fwd_return], axis=1).dropna()
    if len(paired) < 3:
        return float("nan")
    a = paired.iloc[:, 0].to_numpy(dtype=float)
    b = paired.iloc[:, 1].to_numpy(dtype=float)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    # scipy.stats.spearmanr returns a SignificanceResult namedtuple
    # (correlation, pvalue); index 0 is the coefficient. Tuple-indexing
    # avoids the .statistic attribute access that pandas-stubs can't
    # narrow on the union return.
    result = stats.spearmanr(a, b)
    rho = np.asarray(result[0]).item()
    return float(rho)


def aggregate_snapshot(snapshot_df: pd.DataFrame) -> Dict[str, object]:
    """Per-snapshot aggregation across sectors.

    Within each sector: rank by deviation, form 5 equal-weight
    quintiles, take Q1 (most-negative deviation = cheapest) as the long
    leg, Q5 (most-positive = richest) as the short leg, compute
    `long_ret - short_ret`. Across sectors: equal-weight the per-sector
    long-short returns (sector neutralization — the model is sector-
    relative so cross-sector aggregation must not let one sector
    dominate). Also computes a per-sector Spearman IC and stores the
    cross-sector mean IC.

    Returns a dict with keys snapshot, n_sectors, ls_return (gross),
    mean_ic, per_sector (list of dicts).
    """
    if snapshot_df.empty:
        return {}
    per_sector: List[Dict[str, object]] = []
    for sector, sub in snapshot_df.groupby("Sector", sort=False):
        # qcut with duplicates='drop' protects against ties at the
        # quintile boundaries (a sector where many tickers share a
        # deviation of exactly zero would otherwise raise). Sectors
        # whose deviation distribution collapses to <5 distinct buckets
        # are skipped — there's no meaningful long-short to form.
        try:
            sub = sub.copy()
            sub["quintile"] = pd.qcut(
                sub["deviation"], _N_QUINTILES, labels=False, duplicates="drop"
            )
        except ValueError:
            continue
        if sub["quintile"].nunique() < _N_QUINTILES:
            continue
        long_leg = float(sub[sub["quintile"] == 0]["fwd_return"].mean())
        short_leg = float(sub[sub["quintile"] == _N_QUINTILES - 1]["fwd_return"].mean())
        ls = long_leg - short_leg
        dev_series = sub["deviation"]
        ret_series = sub["fwd_return"]
        assert isinstance(dev_series, pd.Series)
        assert isinstance(ret_series, pd.Series)
        ic = _spearman_ic(dev_series, ret_series)
        per_sector.append(
            {
                "sector": str(sector),
                "n": int(len(sub)),
                "long_return": float(long_leg),
                "short_return": float(short_leg),
                "ls_return": ls,
                "ic": ic,
            }
        )
    if not per_sector:
        return {}
    ls_values: List[float] = [
        float(p["ls_return"]) for p in per_sector if isinstance(p["ls_return"], float)
    ]
    ls_agg = float(np.mean(ls_values)) if ls_values else float("nan")
    ic_values: List[float] = [
        float(p["ic"])
        for p in per_sector
        if isinstance(p["ic"], float) and not np.isnan(p["ic"])
    ]
    ic_agg = float(np.mean(ic_values)) if ic_values else float("nan")
    return {
        "snapshot": snapshot_df["snapshot"].iloc[0],
        "n_sectors": len(per_sector),
        "ls_return": ls_agg,
        "mean_ic": ic_agg,
        "per_sector": per_sector,
    }


def run_backtest(
    months: int = 36,
    cost_bps: float = 10.0,
    sa: Optional[pd.DataFrame] = None,
    prices: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, BacktestMetrics]:
    """End-to-end backtest. Returns (monthly_df, per_sector_df, metrics).

    ``cost_bps`` is applied as a flat round-trip transaction cost on the
    long-short return at every rebalance (Q1 long + Q5 short turns over
    each month; ``cost_bps`` deducts the full round trip from the
    gross return). ``sa`` and ``prices`` are injectable for testing;
    when omitted they're loaded / fetched from disk.
    """
    if sa is None:
        sa = load_sector_analysis()
    if prices is None:
        tickers = sa["Ticker"].astype(str).unique().tolist()
        # Fetch a buffer of months+3 to give build_snapshot_dates room
        # to enforce the t+1mo forward-window constraint cleanly.
        end = pd.Timestamp.today().normalize()
        start = end - pd.DateOffset(months=months + 3)
        prices = fetch_prices(tickers, start, end)

    assert isinstance(prices.index, pd.DatetimeIndex)
    snaps = build_snapshot_dates(prices.index, months=months)
    if not snaps:
        raise RuntimeError(
            f"No valid snapshot dates in price index covering {months} months."
        )

    panel = build_snapshot_panel(sa, prices)

    monthly_rows: List[Dict[str, object]] = []
    per_sector_rows: List[Dict[str, object]] = []
    for snapshot_date in snaps:
        forward_date = _next_trading_day(
            prices.index, snapshot_date + pd.DateOffset(months=_FORWARD_MONTHS)
        )
        if forward_date is None:
            continue
        sig = compute_snapshot_signal(panel, prices, snapshot_date, forward_date)
        agg = aggregate_snapshot(sig)
        if not agg:
            continue
        monthly_rows.append(
            {
                "snapshot": agg["snapshot"],
                "forward": forward_date,
                "n_sectors": agg["n_sectors"],
                "ls_return_gross": agg["ls_return"],
                "mean_ic": agg["mean_ic"],
            }
        )
        per_sector_value = agg["per_sector"]
        if not isinstance(per_sector_value, list):
            continue
        for entry in per_sector_value:
            row = dict(entry)
            row["snapshot"] = agg["snapshot"]
            per_sector_rows.append(row)

    if not monthly_rows:
        raise RuntimeError("Backtest produced zero usable snapshots.")

    monthly_df = pd.DataFrame(monthly_rows).sort_values("snapshot").reset_index(drop=True)
    per_sector_df = pd.DataFrame(per_sector_rows).sort_values(
        ["snapshot", "sector"]
    ).reset_index(drop=True)

    # Apply round-trip transaction cost. 10 bps per round trip = 0.0010
    # subtracted from each month's gross long-short return. We treat
    # the strategy as fully turning over each month (a defensible
    # worst-case for a monthly rebalance — in practice some names
    # persist across rebalances, but quantifying that requires a name-
    # level turnover tracker we deliberately keep out of scope).
    cost_decimal = cost_bps / 10_000.0
    monthly_df["ls_return_net"] = monthly_df["ls_return_gross"] - cost_decimal
    monthly_df["cumulative_net"] = (1.0 + monthly_df["ls_return_net"]).cumprod()

    metrics = _compute_metrics(monthly_df, per_sector_df, months, cost_bps)
    return monthly_df, per_sector_df, metrics


def _compute_metrics(
    monthly_df: pd.DataFrame,
    per_sector_df: pd.DataFrame,
    months: int,
    cost_bps: float,
) -> BacktestMetrics:
    """Roll up monthly returns + per-sector ICs into top-line metrics."""
    net_ret = monthly_df["ls_return_net"].astype(float)
    mean_ret = float(net_ret.mean())
    std_ret = float(net_ret.std(ddof=1)) if len(net_ret) > 1 else 0.0
    sharpe = (
        mean_ret / std_ret * float(np.sqrt(_MONTHS_PER_YEAR))
        if std_ret > 0
        else float("nan")
    )
    cum_ret = float(monthly_df["cumulative_net"].iloc[-1]) - 1.0
    cum_series = monthly_df["cumulative_net"].astype(float)
    drawdown = cum_series / cum_series.cummax() - 1.0
    max_dd = float(drawdown.min())
    hit_rate = float((net_ret > 0).mean())

    # IC stats: take the per-sector IC observations (sector-snapshot
    # pairs) as the population. Mean / t-stat / IR are standard
    # cross-sectional-model conventions.
    ic_obs = per_sector_df["ic"].astype(float).dropna()
    mean_ic = float(ic_obs.mean()) if len(ic_obs) else float("nan")
    ic_std = float(ic_obs.std(ddof=1)) if len(ic_obs) > 1 else float("nan")
    ic_t = (
        mean_ic / (ic_std / float(np.sqrt(len(ic_obs))))
        if len(ic_obs) > 1 and ic_std > 0
        else float("nan")
    )
    # Information ratio uses the per-snapshot cross-sector mean IC
    # series so it's a proper time-series ratio (matches Grinold's IR
    # convention), not a within-sector ratio.
    snapshot_ic = monthly_df["mean_ic"].astype(float).dropna()
    if len(snapshot_ic) > 1 and snapshot_ic.std(ddof=1) > 0:
        ic_ir = float(
            snapshot_ic.mean()
            / snapshot_ic.std(ddof=1)
            * float(np.sqrt(_MONTHS_PER_YEAR))
        )
    else:
        ic_ir = float("nan")

    return BacktestMetrics(
        months=months,
        snapshots=len(monthly_df),
        cost_bps=cost_bps,
        mean_ic=mean_ic,
        ic_t_stat=ic_t,
        ic_information_ratio=ic_ir,
        ls_mean_monthly_return=mean_ret,
        ls_std_monthly_return=std_ret,
        ls_sharpe_annualized=sharpe,
        ls_cumulative_return=cum_ret,
        ls_max_drawdown=max_dd,
        ls_hit_rate=hit_rate,
    )


def cost_sensitivity(
    monthly_df: pd.DataFrame, cost_bps_list: List[float]
) -> pd.DataFrame:
    """Apply each cost level to gross returns and report Sharpe / CumRet.

    Recomputes net returns from the gross series (`ls_return_gross`)
    rather than scaling the already-net series, so the table is
    independent of the cost_bps the run was configured with.
    """
    gross = monthly_df["ls_return_gross"].astype(float)
    rows: List[Dict[str, float]] = []
    for cost_bps in cost_bps_list:
        cost = cost_bps / 10_000.0
        net = gross - cost
        std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        sharpe = (
            float(net.mean()) / std * float(np.sqrt(_MONTHS_PER_YEAR))
            if std > 0
            else float("nan")
        )
        compounded = (net + 1.0).prod()
        cum = float(compounded) - 1.0
        rows.append(
            {
                "cost_bps": float(cost_bps),
                "mean_monthly_return": float(net.mean()),
                "sharpe_annualized": sharpe,
                "cumulative_return": cum,
                "hit_rate": float((net > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def sector_summary(per_sector_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sector IC / LS-Sharpe summary across all snapshots.

    Returns one row per sector with: snapshots observed, mean IC,
    IC t-stat, long-short mean monthly return, long-short annualized
    Sharpe, long-short hit rate. Sectors that never accumulated enough
    snapshots to compute a t-stat have NaN there but still report the
    mean.
    """
    rows: List[Dict[str, object]] = []
    grouped = per_sector_df.groupby("sector", sort=True)
    for sector, sub in grouped:
        ic = sub["ic"].astype(float).dropna()
        ls = sub["ls_return"].astype(float).dropna()
        n_snap = int(len(sub))
        mean_ic = float(ic.mean()) if len(ic) else float("nan")
        ic_t = (
            float(mean_ic / (ic.std(ddof=1) / np.sqrt(len(ic))))
            if len(ic) > 1 and ic.std(ddof=1) > 0
            else float("nan")
        )
        ls_mean = float(ls.mean()) if len(ls) else float("nan")
        ls_std = float(ls.std(ddof=1)) if len(ls) > 1 else float("nan")
        ls_sharpe = (
            ls_mean / ls_std * float(np.sqrt(_MONTHS_PER_YEAR))
            if ls_std and ls_std > 0
            else float("nan")
        )
        hit = float((ls > 0).mean()) if len(ls) else float("nan")
        rows.append(
            {
                "sector": str(sector),
                "n_snapshots": n_snap,
                "mean_ic": mean_ic,
                "ic_t_stat": ic_t,
                "ls_mean_monthly": ls_mean,
                "ls_sharpe": ls_sharpe,
                "ls_hit_rate": hit,
            }
        )
    return pd.DataFrame(rows)


RESULTS_CSV_PATH = ROOT / "backtest_results.csv"
ARTIFACTS_DIR = ROOT / "backtest_artifacts"

# Match the dashboard's COLORS palette (src/dashboard.py:66-74) so the
# backtest artifacts share visual identity with the live UI.
_PLOT_COLORS = {
    "primary": "#3498db",
    "secondary": "#e74c3c",
    "accent": "#2ecc71",
    "text": "#2c3e50",
    "light_gray": "#f0f0f0",
}


def write_results_csv(
    monthly_df: pd.DataFrame,
    per_sector_df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """Persist monthly-level + per-sector results into one wide CSV.

    Each row is a (snapshot, sector) pair so the file captures both the
    per-snapshot aggregate (n_sectors, ls_return_net, mean_ic,
    cumulative_net) and the per-sector breakdown (sector-level IC and
    LS return) in one place. A consumer that only wants the
    cross-sector time series can collapse by snapshot.
    """
    if path is None:
        path = RESULTS_CSV_PATH
    monthly_projection = monthly_df[
        [
            "snapshot",
            "forward",
            "n_sectors",
            "ls_return_gross",
            "ls_return_net",
            "mean_ic",
            "cumulative_net",
        ]
    ]
    assert isinstance(monthly_projection, pd.DataFrame)
    monthly_cols = monthly_projection.rename(
        columns={
            "mean_ic": "snapshot_mean_ic",
            "ls_return_gross": "snapshot_ls_gross",
            "ls_return_net": "snapshot_ls_net",
            "cumulative_net": "snapshot_cumulative_net",
        }
    )
    merged = per_sector_df.merge(monthly_cols, on="snapshot", how="left")
    merged = merged.sort_values(["snapshot", "sector"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return path


def _save_cumulative_plot(monthly_df: pd.DataFrame, path: Path) -> None:
    """Cumulative long-short return curve. Net of the configured cost."""
    # Import lazily so importing the module for unit tests / interactive
    # work doesn't pull matplotlib's backend stack.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        monthly_df["snapshot"],
        monthly_df["cumulative_net"],
        color=_PLOT_COLORS["primary"],
        linewidth=2.0,
        label="Long-short (Q1 - Q5), sector-neutral, net",
    )
    ax.axhline(1.0, color=_PLOT_COLORS["light_gray"], linewidth=1.0, linestyle="--")
    ax.set_xlabel("Snapshot date", color=_PLOT_COLORS["text"])
    ax.set_ylabel("Cumulative net return (x initial)", color=_PLOT_COLORS["text"])
    ax.set_title(
        "Cumulative long-short return", color=_PLOT_COLORS["text"], fontsize=13
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_ic_distribution_plot(per_sector_df: pd.DataFrame, path: Path) -> None:
    """Per-sector IC distribution (box plot of per-snapshot ICs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sectors = sorted(per_sector_df["sector"].unique())
    data = [
        per_sector_df.loc[per_sector_df["sector"] == s, "ic"].dropna().to_numpy()
        for s in sectors
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        data,
        tick_labels=sectors,
        patch_artist=True,
        medianprops={"color": _PLOT_COLORS["secondary"], "linewidth": 1.5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_PLOT_COLORS["primary"])
        patch.set_alpha(0.35)
    ax.axhline(0.0, color=_PLOT_COLORS["text"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("Sector", color=_PLOT_COLORS["text"])
    ax.set_ylabel(
        "Per-snapshot IC (Spearman, deviation vs forward return)",
        color=_PLOT_COLORS["text"],
    )
    ax.set_title("IC distribution by sector", color=_PLOT_COLORS["text"], fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_artifacts(
    monthly_df: pd.DataFrame,
    per_sector_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Save the two PNG artifacts. Returns the saved paths."""
    if out_dir is None:
        out_dir = ARTIFACTS_DIR
    cum_path = out_dir / "cumulative_long_short_return.png"
    ic_path = out_dir / "ic_distribution_by_sector.png"
    _save_cumulative_plot(monthly_df, cum_path)
    _save_ic_distribution_plot(per_sector_df, ic_path)
    return cum_path, ic_path


def format_summary_table(
    metrics: BacktestMetrics,
    per_sector_df: pd.DataFrame,
    cost_table: pd.DataFrame,
) -> str:
    """Pretty-printed summary matching generate_weights.py's style.

    Same column-padding conventions: left-aligned section labels, right-
    aligned numerics with explicit width specifiers, a separator line
    equal to the header length, and a section break between the headline
    table, the per-sector table, and the cost-sensitivity table.
    """
    headline_rows = [
        f"{'metric':<32} {'value':>14}",
        "-" * 47,
        f"{'months':<32} {metrics.months:>14d}",
        f"{'snapshots':<32} {metrics.snapshots:>14d}",
        f"{'cost (bps round-trip)':<32} {metrics.cost_bps:>14.2f}",
        f"{'mean IC (Spearman)':<32} {metrics.mean_ic:>14.4f}",
        f"{'IC t-stat':<32} {metrics.ic_t_stat:>14.3f}",
        f"{'IC information ratio':<32} {metrics.ic_information_ratio:>14.3f}",
        f"{'LS mean monthly return':<32} {metrics.ls_mean_monthly_return:>14.5f}",
        f"{'LS monthly std':<32} {metrics.ls_std_monthly_return:>14.5f}",
        f"{'LS annualized Sharpe':<32} {metrics.ls_sharpe_annualized:>14.3f}",
        f"{'LS cumulative return':<32} {metrics.ls_cumulative_return:>14.4f}",
        f"{'LS max drawdown':<32} {metrics.ls_max_drawdown:>14.4f}",
        f"{'LS hit rate':<32} {metrics.ls_hit_rate:>14.2%}",
    ]

    sector_header = (
        f"{'sector':<24} {'n_snaps':>8} {'mean_IC':>9} {'IC_t':>7} "
        f"{'LS_ret':>9} {'LS_Sharpe':>10} {'LS_hit':>8}"
    )
    sector_rows = [sector_header, "-" * len(sector_header)]
    for _, row in per_sector_df.iterrows():
        sector_rows.append(
            f"{str(row['sector']):<24} "
            f"{int(row['n_snapshots']):>8d} "
            f"{float(row['mean_ic']):>9.4f} "
            f"{float(row['ic_t_stat']):>7.2f} "
            f"{float(row['ls_mean_monthly']):>9.4f} "
            f"{float(row['ls_sharpe']):>10.3f} "
            f"{float(row['ls_hit_rate']):>8.2%}"
        )

    cost_header = (
        f"{'cost_bps':>10} {'monthly_ret':>13} {'Sharpe':>10} "
        f"{'cumret':>10} {'hit_rate':>10}"
    )
    cost_rows = [cost_header, "-" * len(cost_header)]
    for _, row in cost_table.iterrows():
        cost_rows.append(
            f"{float(row['cost_bps']):>10.2f} "
            f"{float(row['mean_monthly_return']):>13.5f} "
            f"{float(row['sharpe_annualized']):>10.3f} "
            f"{float(row['cumulative_return']):>10.4f} "
            f"{float(row['hit_rate']):>10.2%}"
        )

    return (
        "\nBacktest summary:\n"
        + "\n".join(headline_rows)
        + "\n\nPer-sector breakdown:\n"
        + "\n".join(sector_rows)
        + "\n\nCost sensitivity (computed from gross returns):\n"
        + "\n".join(cost_rows)
        + "\n"
    )


# Cost-sensitivity grid: 0 (frictionless upper bound), 10 (the spec's
# round-trip assumption), 25 (a stress test for less-liquid names).
_COST_SENSITIVITY_GRID = [0.0, 10.0, 25.0]


def _parse_args() -> "object":
    """argparse parser kept inline to keep the CLI surface in one file."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the sector-relative valuation backtest. See BACKTEST.md "
            "for methodology and limitations."
        )
    )
    parser.add_argument(
        "--months",
        type=int,
        default=36,
        help="Backtest window length in months (default: 36).",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help=(
            "Round-trip transaction cost in basis points applied to the "
            "long-short return at each rebalance (default: 10)."
        ),
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_CSV_PATH,
        help=f"Path for the per-(snapshot, sector) results CSV "
        f"(default: {RESULTS_CSV_PATH.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help=f"Directory for PNG artifacts "
        f"(default: {ARTIFACTS_DIR.relative_to(ROOT)}).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = _parse_args()
    months = int(getattr(args, "months"))
    cost_bps = float(getattr(args, "cost_bps"))
    results_csv = Path(getattr(args, "results_csv"))
    artifacts_dir = Path(getattr(args, "artifacts_dir"))

    monthly_df, per_sector_df, metrics = run_backtest(
        months=months, cost_bps=cost_bps
    )
    summary = sector_summary(per_sector_df)
    cost_table = cost_sensitivity(monthly_df, _COST_SENSITIVITY_GRID)

    write_results_csv(monthly_df, per_sector_df, results_csv)
    save_artifacts(monthly_df, per_sector_df, out_dir=artifacts_dir)

    print(format_summary_table(metrics, summary, cost_table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
