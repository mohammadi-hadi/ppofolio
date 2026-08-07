"""Empirical reward-balance audit.

Rolls simple reference policies (random, buy-and-hold-ish constant weights) over a
slice of the real dataset at a given config/timeframe and prints the mean |weighted
component| per DAY. Run for both configs and compare rows — no component should
dominate unexpectedly, and per-day masses should be comparable across timeframes.

Usage:
  .venv/bin/python scripts/reward_audit.py --config configs/full.yaml
  .venv/bin/python scripts/reward_audit.py --config configs/hourly.yaml
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppofolio.env.crypto_trading_env import CryptoTradingEnv  # noqa: E402
from ppofolio.utils import ROOT, load_config  # noqa: E402

WEIGHT_KEYS = {"r_return": "return", "r_sharpe": "sharpe", "r_drawdown": "drawdown",
               "r_liquidity": "liquidity", "r_sentiment": "sentiment",
               "r_loss_asymm": "loss_asymm", "r_borrow": "borrow", "r_turnover": "turnover"}


def roll(env: CryptoTradingEnv, policy, n_steps: int, cfg: dict) -> dict:
    obs, _ = env.reset(seed=0)
    sums = defaultdict(float)
    w = cfg["reward"]["weights"]
    bars_per_day = max(int(cfg.get("bars_per_day", 1)), 1)
    steps = 0
    while steps < n_steps:
        action = policy(env)
        obs, r, term, trunc, info = env.step(action)
        for comp, wkey in WEIGHT_KEYS.items():
            base = float(w.get(wkey, 0.0))
            if wkey in ("drawdown", "sentiment", "sharpe"):
                base = base / bars_per_day
            sums[comp] += abs(base * info.get(comp, 0.0))
        sums["total_reward"] += abs(r)
        steps += 1
        if term or trunc:
            obs, _ = env.reset()
    days = steps / bars_per_day
    return {k: v / days for k, v in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(Path(ROOT) / cfg["data_paths"]["processed"])
    panel.index = pd.to_datetime(panel.index, utc=True)
    tr = cfg["splits"]["train"]
    panel = panel.loc[pd.Timestamp(str(tr["start"]), tz="UTC"): pd.Timestamp(str(tr["end"]), tz="UTC")]

    n = len(cfg["assets"])
    rng = np.random.default_rng(0)
    low = -1.0 if cfg["action"]["allow_short"] else 0.0

    policies = {
        "random": lambda env: rng.uniform(low, 1.0, n),
        "hold_equal": lambda env: np.full(n, 1.0 / n),
        "flat": lambda env: np.zeros(n),
    }
    print(f"config={args.config} timeframe={cfg.get('timeframe', '1d')} steps={args.steps}")
    rows = {}
    for name, pol in policies.items():
        env = CryptoTradingEnv(panel, cfg, enforce_horizon=True)
        rows[name] = roll(env, pol, args.steps, cfg)
    df = pd.DataFrame(rows).T
    with pd.option_context("display.float_format", lambda x: f"{x:9.5f}"):
        print("mean |weighted component| per DAY:")
        print(df.to_string())


if __name__ == "__main__":
    main()
