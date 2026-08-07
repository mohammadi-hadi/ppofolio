"""Per-model validation-split evaluation (model selection ONLY — never test).

Rolls each given model zip deterministically over the config's val split and
prints/saves total return, Sharpe, max drawdown vs the fee-adjusted baselines.

Usage:
  .venv/bin/python scripts/eval_val.py --config configs/hourly.yaml \
      --models models/policy_h42.zip models/policy_h7.zip
  .venv/bin/python scripts/eval_val.py --config configs/full.yaml --models 'models/policy_s*.zip'
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppofolio.backtest.baselines import bh_rebalanced, buy_and_hold, macd_crossover  # noqa: E402
from ppofolio.backtest.metrics import from_equity  # noqa: E402
from ppofolio.backtest.run_backtest import roll_policy  # noqa: E402
from ppofolio.agents.ensemble import EnsemblePolicy  # noqa: E402
from ppofolio.utils import ROOT, ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model zip paths or globs; 'ENSEMBLE:<glob>' evaluates them as one ensemble")
    ap.add_argument("--out", default="reports/eval_val.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(Path(ROOT) / cfg["data_paths"]["processed"])
    panel.index = pd.to_datetime(panel.index, utc=True)

    v = cfg["splits"]["val"]
    s = pd.Timestamp(str(v["start"]), tz="UTC")
    e = pd.Timestamp(str(v["end"]), tz="UTC")
    lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    window_start = panel.index[max(panel.index.searchsorted(s) - lookback, 0)]
    slab = panel.loc[window_start:e]

    close_cols = [c for c in panel.columns if c.endswith("__close")]
    prices = panel.loc[s:e, close_cols].copy()
    prices.columns = [c.split("__")[0] for c in prices.columns]

    bpy = int(cfg.get("bars_per_year", 365))
    fee_bps = float(cfg.get("fee_bps", 0.0))
    rows = [
        {"model": "bh_hold", **from_equity(buy_and_hold(prices, cfg["initial_cash"], fee_bps=fee_bps), bpy)},
        {"model": "bh_rebalanced", **from_equity(bh_rebalanced(prices, cfg["initial_cash"], fee_bps=fee_bps), bpy)},
        {"model": "macd", **from_equity(macd_crossover(prices, cfg["initial_cash"], fee_bps=fee_bps), bpy)},
    ]

    for spec in args.models:
        if spec.startswith("ENSEMBLE:"):
            paths = sorted(glob.glob(spec[len("ENSEMBLE:"):]))
            if not paths:
                print(f"[eval_val] no files match {spec}")
                continue
            model = EnsemblePolicy(paths)
            name = "ensemble(" + ",".join(Path(p).stem for p in paths) + ")"
            eq = roll_policy(model, slab, cfg)
            rows.append({"model": name, **from_equity(eq.loc[eq.index >= s], bpy)})
            continue
        for p in sorted(glob.glob(spec)):
            model = PPO.load(p)
            eq = roll_policy(model, slab, cfg)
            rows.append({"model": Path(p).stem, **from_equity(eq.loc[eq.index >= s], bpy)})

    report = pd.DataFrame(rows)
    out = Path(ROOT) / args.out
    ensure_dir(out.parent)
    report.to_csv(out, index=False)
    print(f"val window: {s.date()} .. {e.date()}  fee_bps={fee_bps}")
    print(report.to_string(index=False))
    print(f"[eval_val] wrote {out}")


if __name__ == "__main__":
    main()
