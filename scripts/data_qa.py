"""Dataset QA gate — run before any training. Non-zero exit on failure.

Checks: monotonic unique index, gap ratio, NaNs, positive closes, on-chain
freshness/constancy (warn-only: CoinMetrics community availability varies).

Usage: .venv/bin/python scripts/data_qa.py --config configs/hourly.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppofolio.utils import ROOT, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = Path(ROOT) / cfg["data_paths"]["processed"]
    panel = pd.read_parquet(path)
    panel.index = pd.to_datetime(panel.index, utc=True)
    timeframe = cfg.get("timeframe", "1d")
    freq = pd.Timedelta(hours=1) if timeframe == "1h" else pd.Timedelta(days=1)

    failures: list[str] = []
    warnings: list[str] = []

    if not panel.index.is_monotonic_increasing:
        failures.append("index not monotonic")
    if panel.index.has_duplicates:
        failures.append("index has duplicates")

    expected = int((panel.index.max() - panel.index.min()) / freq) + 1
    gap_ratio = 1.0 - len(panel) / max(expected, 1)
    if gap_ratio > 0.001:
        failures.append(f"gap ratio {gap_ratio:.4%} > 0.1%")

    n_nan = int(panel.isna().sum().sum())
    if n_nan > 0:
        failures.append(f"{n_nan} NaNs after finalize")

    close_cols = [c for c in panel.columns if c.endswith("__close")]
    if (panel[close_cols] <= 0).any().any():
        failures.append("non-positive closes present")

    # On-chain freshness/constancy — warn-only (community API availability varies).
    for c in [c for c in panel.columns if any(c.endswith(f"__{o}") for o in cfg["features"]["onchain"])]:
        tail = panel[c].tail(30 if timeframe == "1d" else 30 * 24)
        if tail.nunique() <= 1:
            warnings.append(f"on-chain feature {c} constant over the last 30 days (stale/zero-filled)")

    print(f"dataset: {path}")
    print(f"rows={len(panel)} cols={panel.shape[1]} range=[{panel.index.min()} .. {panel.index.max()}]")
    print(f"gap_ratio={gap_ratio:.5%} nans={n_nan}")
    for w in warnings:
        print(f"WARN: {w}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("QA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
