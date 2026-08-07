"""Backtest: run the trained policy deterministically over each regime slice + baselines."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ppofolio.agents.ensemble import load_ensemble
from ppofolio.backtest.baselines import (bh_rebalanced, buy_and_hold, macd_crossover,
                                    risk_parity, tsmom_ma200)
from ppofolio.backtest.metrics import from_equity
from ppofolio.env.crypto_trading_env import CryptoTradingEnv
from ppofolio.utils import ROOT, ensure_dir, load_config


def roll_policy(model, panel: pd.DataFrame, cfg: dict) -> pd.Series:
    lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    env = CryptoTradingEnv(panel, cfg, episode_start_idx=lookback, enforce_horizon=False)
    obs, _ = env.reset(seed=0)
    # Mirror the live loop's optional turnover damping so backtests measure the
    # execution the service would actually run.
    alpha = float(cfg.get("action", {}).get("target_smoothing", 0.0))
    equity = [env.equity]
    dates = [panel.index[env._t]]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if alpha > 0:
            action = alpha * env.weights + (1.0 - alpha) * np.asarray(action, dtype=float).reshape(-1)
        obs, _, term, trunc, info = env.step(action)
        equity.append(info["equity"])
        dates.append(panel.index[env._t])
        done = term or trunc
    return pd.Series(equity, index=pd.to_datetime(dates), name="equity")


def evaluate_windows(model, panel: pd.DataFrame, cfg: dict,
                     windows: dict[str, tuple[str, str]], out_name: str) -> pd.DataFrame:
    results: list[dict] = []
    close_cols = [c for c in panel.columns if c.endswith("__close")]
    prices_all = panel[close_cols].copy()
    prices_all.columns = [c.split("__")[0] for c in prices_all.columns]

    bpy = int(cfg.get("bars_per_year", 365))
    fee_bps = float(cfg.get("fee_bps", 0.0))
    for regime, (s, e) in windows.items():
        s = pd.Timestamp(str(s), tz="UTC")
        e = pd.Timestamp(str(e), tz="UTC")
        # include lookback buffer before s for the env
        lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
        window_start = panel.index[max(panel.index.searchsorted(s) - lookback, 0)]
        slab = panel.loc[window_start:e]
        prices = prices_all.loc[s:e]

        hist = prices_all.loc[:e]  # warmup history for vol/MA windows
        bh = buy_and_hold(prices, cfg["initial_cash"], fee_bps=fee_bps)
        bhr = bh_rebalanced(prices, cfg["initial_cash"], fee_bps=fee_bps)
        macd = macd_crossover(prices, cfg["initial_cash"], fee_bps=fee_bps)
        rp = risk_parity(prices, cfg["initial_cash"], fee_bps=fee_bps, history=hist)
        tsm = tsmom_ma200(prices, cfg["initial_cash"], fee_bps=fee_bps, history=hist)
        results.append({"regime": regime, "strategy": "bh_hold", **from_equity(bh, bpy)})
        results.append({"regime": regime, "strategy": "bh_rebalanced", **from_equity(bhr, bpy)})
        results.append({"regime": regime, "strategy": "macd", **from_equity(macd, bpy)})
        results.append({"regime": regime, "strategy": "risk_parity", **from_equity(rp, bpy)})
        results.append({"regime": regime, "strategy": "tsmom_ma200", **from_equity(tsm, bpy)})

        if model is not None:
            try:
                eq = roll_policy(model, slab, cfg)
                eq = eq.loc[eq.index >= s]
                results.append({"regime": regime, "strategy": "ppo", **from_equity(eq, bpy)})
            except Exception:
                # A silent miss here once hid an obs-shape mismatch for weeks.
                traceback.print_exc()
                raise

    report = pd.DataFrame(results)
    out_dir = ensure_dir(Path(ROOT) / "reports")
    report.to_csv(out_dir / out_name, index=False)
    print(report.to_string(index=False))
    print(f"[backtest] wrote {out_dir / out_name}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default="models/policy.zip")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="'test' = regime windows in splits.test_regimes; 'val' = the val split")
    ap.add_argument("--window", nargs=2, default=None, metavar=("START", "END"),
                    help="explicit date window; overrides --split")
    ap.add_argument("--label", default="window", help="regime label for --window runs")
    ap.add_argument("--out", default=None, help="report filename (default backtest.csv / backtest_val.csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(Path(ROOT) / cfg["data_paths"]["processed"])
    panel.index = pd.to_datetime(panel.index, utc=True)

    ensemble = load_ensemble(Path(ROOT) / "models", prefix=cfg.get("model_prefix", "policy_s"),
                             disagreement=(cfg.get("ensemble", {}) or {}).get("disagreement_scaling"))
    model_path = Path(ROOT) / args.model
    if ensemble is not None:
        model = ensemble
        print(f"[backtest] using ensemble of {len(ensemble.models)} models")
    elif model_path.exists():
        model = PPO.load(model_path)
        print(f"[backtest] single model at {model_path}")
    else:
        model = None
        print("[backtest] no model found, baselines only")

    if args.window is not None:
        windows = {args.label: (args.window[0], args.window[1])}
        out_name = args.out or f"backtest_{args.label}.csv"
    elif args.split == "val":
        v = cfg["splits"]["val"]
        windows = {"val": (str(v["start"]), str(v["end"]))}
        out_name = args.out or "backtest_val.csv"
    else:
        windows = {k: (str(s), str(e)) for k, (s, e) in cfg["splits"]["test_regimes"].items()}
        out_name = args.out or "backtest.csv"

    evaluate_windows(model, panel, cfg, windows, out_name)


if __name__ == "__main__":
    main()
