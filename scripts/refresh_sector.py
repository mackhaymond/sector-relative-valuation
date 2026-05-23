#!/usr/bin/env python3
"""Refresh one sector's data and merge it into the existing CSVs.

Usage::

    uv run python scripts/refresh_sector.py technology

Use this to patch a single sector when a full GHA refresh succeeded for
the other sectors but lost one to a transient yfinance failure. Avoids
re-running the full ~100-minute pipeline.

Steps performed:

1. Run the existing ``data.process_sector_async`` for the named sector
   (uses the same rate-limit settings as the full pipeline).
2. Read the existing ``sector_analysis.csv`` and ``sector_analysis_full.csv``.
3. Drop any rows for the named sector from both CSVs.
4. Append the freshly-fetched rows.
5. Sort by (Sector, Ticker) for a deterministic on-disk order.
6. Write the merged CSVs back.
7. Shell out to ``src/generate_weights.py`` to regenerate ``weights.csv``
   against the merged data.

Exits 1 if the sector still produces zero rows so the operator notices.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data as data_module  # noqa: E402

SIMPLE_CSV = ROOT / "sector_analysis.csv"
FULL_CSV = ROOT / "sector_analysis_full.csv"


async def _refresh(sector: str) -> None:
    if sector not in data_module.SECTORS:
        sys.exit(f"error: unknown sector {sector!r}; valid: {data_module.SECTORS}")

    print(f"Fetching {sector!r} from yfinance...", flush=True)
    full_df, simple_df = await data_module.process_sector_async(sector, data_module.METRICS)
    if simple_df is None or simple_df.empty:
        sys.exit(f"error: refresh yielded 0 rows for {sector!r}")
    if full_df is None or full_df.empty:
        sys.exit(f"error: refresh yielded no detailed-CSV rows for {sector!r}")

    print(f"  got {len(simple_df)} surviving rows", flush=True)

    existing_simple = pd.read_csv(SIMPLE_CSV)
    existing_full = pd.read_csv(FULL_CSV)

    merged_simple = pd.concat(
        [existing_simple[existing_simple["Sector"] != sector], simple_df],
        ignore_index=True,
    ).sort_values(["Sector", "Ticker"], kind="stable").reset_index(drop=True)
    merged_full = pd.concat(
        [existing_full[existing_full["Sector"] != sector], full_df],
        ignore_index=True,
    ).sort_values(["Sector", "Ticker"], kind="stable").reset_index(drop=True)

    merged_simple.to_csv(SIMPLE_CSV, index=False)
    merged_full.to_csv(FULL_CSV, index=False)
    print(f"Merged. {SIMPLE_CSV.name}: {len(merged_simple)} rows, "
          f"{FULL_CSV.name}: {len(merged_full)} rows.", flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <sector-slug>")
    sector = sys.argv[1]

    asyncio.run(_refresh(sector))

    print("Regenerating weights.csv against merged data...", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "generate_weights.py")],
        cwd=str(ROOT),
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
