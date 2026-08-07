"""System-level backtest: policy + circuit breaker + kill switch, as deployed.

The plain backtest rolls the raw policy. The deployed service additionally runs
the volatility circuit breaker and the kill switch (halt to cash on -5%/day or
-15% peak drawdown, no auto-resume). This script rolls a model over a window with
that safety stack simulated, so the numbers describe the SYSTEM that trades.

Usage:
  .venv/bin/python scripts/eval_system.py --config configs/full.yaml \
      --models 'models/gate/policy_s*.zip' --window 2026-02-15 2026-07-10 --label candidate
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppofolio.agents.ensemble import EnsemblePolicy  # noqa: E402
from ppofolio.backtest.metrics import from_equity  # noqa: E402
from ppofolio.env.crypto_trading_env import CryptoTradingEnv  # noqa: E402
from ppofolio.live.circuit_breaker import vol_scale  # noqa: E402
from ppofolio.live.kill_switch import basket_recovery_ok  # noqa: E402
from ppofolio.live.risk_limits import apply_limits  # noqa: E402
from ppofolio.utils import ROOT, ensure_dir, load_config  # noqa: E402


def set_by_path(cfg: dict, dotted: str, raw: str) -> None:
    """Set cfg['a']['b']=value for 'a.b=value' overrides; value parsed as YAML scalar."""
    import yaml
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)


def roll_system(model, panel: pd.DataFrame, cfg: dict, resume_policy: str = "timer",
                recovery_window: int = 10, limits_mode: bool = False, seed: int = 0,
                trace: list | None = None) -> tuple[pd.Series, dict]:
    """resume_policy: 'timer' (48h cooldown, as shipped), 'recovery' (cooldown AND
    basket back above its trailing MA), 'never' (halt is permanent), 'off' (no
    kill switch at all — the raw system).

    limits_mode simulates the deployed execution order harness-side:
    clip/renorm -> rebalance band vs current weights -> vol scale -> invest limits,
    with the env's own band disabled so execution is applied exactly once."""
    lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    env_cfg = cfg
    thr = float(cfg.get("action", {}).get("rebalance_threshold", 0.0))
    if limits_mode:
        env_cfg = {**cfg, "action": {**cfg["action"], "rebalance_threshold": 0.0}}
    env = CryptoTradingEnv(panel, env_cfg, episode_start_idx=lookback, enforce_horizon=False)
    obs, _ = env.reset(seed=seed)

    ks = cfg.get("kill_switch", {})
    max_daily = float(ks.get("max_daily_loss", 0.05))
    max_peak = float(ks.get("max_drawdown_peak", 0.15))
    window_bars = max(int(round(float(ks.get("window_hours", 24.0)) * env.bars_per_day / 24.0)), 1)
    resume_bars = 0
    if float(ks.get("auto_resume_hours", 0.0)) > 0:
        resume_bars = max(int(round(float(ks["auto_resume_hours"]) * env.bars_per_day / 24.0)), 1)

    cb = cfg.get("circuit_breaker", {}) or {}
    btc_col = env.close_cols[0]
    btc = panel[btc_col].to_numpy(dtype=float)

    equity = [env.equity]
    dates = [panel.index[env._t]]
    halted = False
    halted_bars = 0
    peak_base = 0  # index into equity[] from which the running peak is measured
    halts: list[dict] = []
    n_scaled = 0
    gross_traded = 0.0
    fees_paid = 0.0
    low = -1.0 if cfg.get("action", {}).get("allow_short") else 0.0
    max_lev = float(cfg.get("action", {}).get("max_leverage", 1.0))
    closes_all = panel[env.close_cols].to_numpy(dtype=float)
    done = False
    while not done:
        if halted and resume_policy != "never" and resume_bars and halted_bars >= resume_bars:
            resume = True
            if resume_policy == "recovery":
                rec = basket_recovery_ok(closes_all[: env._t + 1], window=recovery_window)
                resume = rec is not False
            if resume:
                halted = False
                peak_base = len(equity) - 1  # re-base the peak at resume
        raw = post_band = None
        scale = 1.0
        if halted:
            halted_bars += 1
            action = np.zeros(env.n)
        else:
            action, _ = model.predict(obs, deterministic=True)
            raw = np.asarray(action, dtype=float).reshape(-1)
            scale, _reason = vol_scale(
                btc[: env._t + 1],
                window=int(cb.get("window", 20)),
                threshold_high=float(cb.get("threshold_high", 0.06)),
                threshold_mid=float(cb.get("threshold_mid", 0.045)),
            )
            if scale < 1.0:
                n_scaled += 1
            if limits_mode:
                target = np.clip(raw, low, 1.0)
                g = np.abs(target).sum()
                if g > max_lev and g > 0:
                    target = target * (max_lev / g)
                if thr > 0:
                    small = np.abs(target - env.weights) < thr
                    target = np.where(small, env.weights, target)
                post_band = target.copy()
                target = target * scale
                price_now = closes_all[env._t]
                target, _notes = apply_limits(target, env.holdings * price_now, env.equity, cfg)
                action = target
            else:
                action = raw * scale
        obs, _, term, trunc, info = env.step(action)
        equity.append(info["equity"])
        dates.append(panel.index[env._t])
        gross_traded += float(info.get("traded_notional", 0.0))
        fees_paid += float(info.get("fees_paid", 0.0))
        if trace is not None:
            trace.append({
                "date": str(dates[-1]), "halted": halted, "scale": scale,
                **{f"raw_{i}": (None if raw is None else float(raw[i])) for i in range(env.n)},
                **{f"band_{i}": (None if post_band is None else float(post_band[i])) for i in range(env.n)},
                **{f"tgt_{i}": float(np.asarray(action).reshape(-1)[i]) for i in range(env.n)},
                **{f"w_{i}": float(info["weights"][i]) for i in range(env.n)},
                "traded_notional": float(info.get("traded_notional", 0.0)),
                "fees": float(info.get("fees_paid", 0.0)),
            })
        done = term or trunc

        if not halted and resume_policy != "off" and ks.get("enabled", True):
            prev = equity[-window_bars - 1] if len(equity) > window_bars else equity[0]
            peak = max(equity[peak_base:])
            daily_loss = (equity[-1] - prev) / max(prev, 1e-9)
            peak_loss = (equity[-1] - peak) / max(peak, 1e-9)
            if daily_loss <= -max_daily or peak_loss <= -max_peak:
                halted = True
                halted_bars = 0
                halts.append({"ts": str(dates[-1]),
                              "reason": f"daily={daily_loss:+.2%} peak={peak_loss:+.2%}"})
    eq = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
    return eq, {"halted": halted, "n_halts": len(halts), "halts": halts, "bars_vol_scaled": n_scaled,
                "gross_traded": gross_traded, "fees_paid": fees_paid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True, help="glob of ensemble member zips")
    ap.add_argument("--window", nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--label", default="system")
    ap.add_argument("--resume-policy", choices=["timer", "recovery", "never", "off"], default="timer")
    ap.add_argument("--recovery-window", type=int, default=10)
    ap.add_argument("--limits", action="store_true",
                    help="simulate deployed execution: band vs weights -> vol scale -> invest limits")
    ap.add_argument("--trace", action="store_true", help="write per-bar trace CSV")
    ap.add_argument("--seed", type=int, default=0, help="slippage RNG seed")
    ap.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE",
                    help="config override, e.g. --set action.rebalance_threshold=0.05")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for override in args.set:
        key, _, val = override.partition("=")
        set_by_path(cfg, key, val)
    panel = pd.read_parquet(Path(ROOT) / cfg["data_paths"]["processed"])
    panel.index = pd.to_datetime(panel.index, utc=True)

    s = pd.Timestamp(args.window[0], tz="UTC")
    e = pd.Timestamp(args.window[1], tz="UTC")
    lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    window_start = panel.index[max(panel.index.searchsorted(s) - lookback, 0)]
    slab = panel.loc[window_start:e]

    paths = sorted(glob.glob(args.models))
    if not paths:
        raise SystemExit(f"no models match {args.models}")
    model = EnsemblePolicy(paths)

    trace_rows: list | None = [] if args.trace else None
    eq, meta = roll_system(model, slab, cfg, resume_policy=args.resume_policy,
                           recovery_window=args.recovery_window, limits_mode=args.limits,
                           seed=args.seed, trace=trace_rows)
    eq = eq.loc[eq.index >= s]
    m = from_equity(eq, int(cfg.get("bars_per_year", 365)))
    fees = meta["fees_paid"]
    gross_pnl = (float(eq.iloc[-1]) - float(eq.iloc[0])) + fees  # price P&L before fees
    fees_pct = fees / abs(gross_pnl) if abs(gross_pnl) > 1e-9 else float("nan")
    row = {"label": args.label, "resume_policy": args.resume_policy,
           "members": ",".join(Path(p).stem for p in paths), **m, **{
        "halted_at_end": meta["halted"], "n_halts": meta["n_halts"],
        "halts": str(meta["halts"]), "bars_vol_scaled": meta["bars_vol_scaled"],
        "gross_traded": round(meta["gross_traded"], 2), "fees_paid": round(fees, 2),
        "fees_pct_gross_pnl": round(fees_pct, 4), "limits": args.limits, "seed": args.seed}}
    print(pd.DataFrame([row]).to_string(index=False))
    out = ensure_dir(Path(ROOT) / "reports") / f"system_{args.label}.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[eval_system] wrote {out}")
    if trace_rows is not None:
        tout = ensure_dir(Path(ROOT) / "reports") / f"trace_{args.label}.csv"
        pd.DataFrame(trace_rows).to_csv(tout, index=False)
        print(f"[eval_system] wrote {tout}")


if __name__ == "__main__":
    main()
