"""Daily live loop (paper mode).

Fetches today's bar via ccxt + on-chain/news, assembles features, runs the policy,
rebalances via PaperBroker.  Designed to be run once per day (or via APScheduler).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ppofolio.agents.ensemble import load_ensemble
from ppofolio.data import coinmetrics, macro, news, ohlcv
from ppofolio.features.assemble import build_panel, finalize
from ppofolio.features.technical import add_technical
from ppofolio.live.circuit_breaker import vol_scale
from ppofolio.live.kill_switch import basket_recovery_ok, check_and_update, is_halted
from ppofolio.live.paper_broker import PaperBroker
from ppofolio.live.risk_limits import apply_limits
from ppofolio.utils import ROOT, ensure_dir, load_config


def fetch_latest_panel(cfg: dict, lookback_bars: int | None = None) -> pd.DataFrame:
    timeframe = cfg.get("timeframe", "1d")
    lookback_bars = int(lookback_bars or cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    # Extra warmup bars for 200-bar indicators; enough for both daily and hourly.
    warmup = 220
    # End on the current bar boundary (exclusive) so the panel only contains CLOSED
    # bars — the model was trained on closed bars, and the last row is also the fill
    # price. The old end=now+1day pulled in the exchange's still-forming candle.
    now = pd.Timestamp.now(tz="UTC")
    if timeframe == "1h":
        end = now.floor("1h")
        start = end - pd.Timedelta(hours=lookback_bars + warmup + 24)
    else:
        end = now.floor("1D")
        start = end - pd.Timedelta(days=lookback_bars + warmup)
    bars_per_day = int(cfg.get("bars_per_day", 1))
    per_asset: dict[str, pd.DataFrame] = {}
    for a in cfg["assets"]:
        slug = a["symbol"].split("/")[0].lower()
        px = ohlcv.load_or_fetch(a["symbol"], start.isoformat(), end.isoformat(),
                                 cfg["data_paths"]["ohlcv"], timeframe=timeframe)
        if timeframe != "1d":
            # Regime features need ~380 daily closes (ma200 + warmup) — far more
            # history than the hourly panel carries. Fetch a daily series for them.
            d_end = now.floor("1D")
            d_start = d_end - pd.Timedelta(days=410)
            dpx = ohlcv.load_or_fetch(a["symbol"], d_start.isoformat(), d_end.isoformat(),
                                      cfg["data_paths"]["ohlcv"], timeframe="1d")
            px = add_technical(px, bars_per_day=bars_per_day, daily_close=dpx["close"])
        else:
            px = add_technical(px)
        oc = coinmetrics.load_or_fetch(a["coinmetrics"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), cfg["data_paths"]["onchain"])
        if timeframe != "1d" and not oc.empty:
            oc = macro.broadcast_to_index(oc, px.index)
        if cfg.get("news", {}).get("enabled"):
            q = cfg["news"]["queries"][[x["symbol"].split("/")[0].lower() for x in cfg["assets"]].index(slug)]
            sn = news.load_or_fetch(q, slug, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), cfg["data_paths"]["news"], cfg["news"]["max_per_day"])
        else:
            idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
            sn = pd.DataFrame(index=idx, data={"sent_mean": 0.0, "sent_count": 0})
        per_asset[slug.upper()] = px.join(oc, how="left").join(sn, how="left")

    panel = build_panel(
        per_asset,
        cfg["features"]["technical"],
        cfg["features"]["onchain"],
        cfg["features"]["sentiment"],
    )
    try:
        m = macro.load_or_fetch_macro(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                                      cfg["data_paths"].get("macro", "data/macro"))
        if timeframe != "1d" and not m.empty:
            m = macro.broadcast_to_index(m, panel.index)
        m.columns = [f"MACRO__{c}" for c in m.columns]
        panel = panel.join(m, how="left")
    except Exception as e:
        print(f"[live] macro fetch failed: {e}")
    return finalize(panel)


def _observation_from_panel(panel: pd.DataFrame, cfg: dict, weights: np.ndarray, cash_frac: float) -> np.ndarray:
    lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    assets = [a["symbol"].split("/")[0] for a in cfg["assets"]]
    tech = cfg["features"]["technical"]
    onc = cfg["features"]["onchain"]
    sen = cfg["features"]["sentiment"]
    macro_feats = cfg["features"].get("macro", [])
    window = panel.iloc[-lookback:]
    parts = []
    for a in assets:
        cols = [f"{a}__{c}" for c in tech + onc + sen]
        parts.append(window[cols].to_numpy(dtype=np.float32))
    asset_block = np.concatenate(parts, axis=1)  # (lookback, n*n_feat)
    if macro_feats:
        macro_cols = [f"MACRO__{c}" for c in macro_feats]
        for c in macro_cols:
            if c not in window.columns:
                window = window.assign(**{c: 0.0})
        macro_block = window[macro_cols].to_numpy(dtype=np.float32)
        block = np.concatenate([asset_block, macro_block], axis=1)
    else:
        block = asset_block
    feats = block.flatten()
    return np.concatenate([feats, weights.astype(np.float32), [np.float32(cash_frac)]]).astype(np.float32).clip(-50, 50)


def run_once(cfg: dict, model_path: Path, broker: PaperBroker) -> dict:
    lookback = int(cfg.get("lookback_bars", cfg.get("lookback_days", 30)))
    panel = fetch_latest_panel(cfg, lookback + 30)
    assets = [a["symbol"].split("/")[0] for a in cfg["assets"]]
    close_cols = [f"{a}__close" for a in assets]
    price = panel[close_cols].iloc[-1].to_numpy(dtype=float)

    holdings = broker.holdings()
    equity = broker.equity(price)
    weights = (holdings * price) / max(equity, 1e-9)
    cash_frac = broker.cash / max(equity, 1e-9)

    # Executions are stamped at wall-clock time (fills happen NOW at the last
    # closed bar's price) — bar timestamps made equity rows sort before same-day
    # reset baselines and broke the time-based kill-switch window.
    ts = int(pd.Timestamp.now(tz="UTC").timestamp())

    # Kill switch: check current equity BEFORE trading. If tripped, force cash.
    ks_cfg = cfg.get("kill_switch", {}) or {}
    recovery_ok = basket_recovery_ok(
        panel[close_cols].to_numpy(dtype=float),
        window=int(ks_cfg.get("recovery_ma_bars", 10)),
    )
    halt_now, halt_reason = check_and_update(broker, equity, cfg, recovery_ok=recovery_ok)
    if halt_now or is_halted(broker):
        target = np.zeros(len(assets))
        gross = float(np.abs(holdings * price).sum())
        if gross > 1.0:
            result = broker.rebalance(target, price, ts=ts)
        else:
            # Already flat: don't re-trade a zero book every run (dust spam).
            result = {"equity": equity, "cash": broker.cash, "weights": weights.tolist(), "fees": 0.0}
        result.update({
            "target_weights": target.tolist(), "price": price.tolist(),
            "assets": assets, "halted": True, "halt_reason": halt_reason,
        })
        return result

    obs = _observation_from_panel(panel, cfg, weights, cash_frac)
    if not np.isfinite(obs).all():
        return {"error": "non-finite observation — run aborted", "equity": equity,
                "weights": weights.tolist(), "halted": is_halted(broker)}
    # Prefer ensemble if policy_s*.zip files exist, else fall back to single policy.zip.
    ens = load_ensemble(Path(ROOT) / "models", prefix=cfg.get("model_prefix", "policy_s"),
                        disagreement=(cfg.get("ensemble", {}) or {}).get("disagreement_scaling"))
    if ens is not None:
        action, _ = ens.predict(obs, deterministic=True)
    else:
        model = PPO.load(model_path)
        action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=float).reshape(-1)
    if not np.isfinite(action).all():
        return {"error": "non-finite action — run aborted", "equity": equity,
                "weights": weights.tolist(), "halted": is_halted(broker)}

    low = -1.0 if cfg["action"]["allow_short"] else 0.0
    target = np.clip(action, low, 1.0)
    gross = np.abs(target).sum()
    if gross > cfg["action"]["max_leverage"] and gross > 0:
        target = target * (cfg["action"]["max_leverage"] / gross)

    # Optional turnover damping: blend the model's target with current weights.
    # 0.0 = trade straight to target (training-env behavior).
    alpha = float(cfg["action"].get("target_smoothing", 0.0))
    if alpha > 0:
        target = alpha * weights + (1.0 - alpha) * target

    # Skip sub-threshold weight changes, mirroring the training env's trade_mask.
    # This runs BEFORE vol scaling and invest limits: the band must judge the
    # model's desired move, not the post-truncation one — with a $250k per-trade
    # cap a desired move is truncated to <=25% weight, and any band above that
    # would silently freeze the book forever.
    thr = float(cfg["action"].get("rebalance_threshold", 0.0))
    threshold_skipped: list[str] = []
    if thr > 0:
        small = np.abs(target - weights) < thr
        threshold_skipped = [assets[i] for i in np.where(small)[0] if abs(target[i] - weights[i]) > 1e-12]
        target = np.where(small, weights, target)

    # Runtime circuit-breaker: scale down exposure if realized vol is high.
    btc_idx = assets.index("BTC") if "BTC" in assets else 0
    btc_closes = panel[f"{assets[btc_idx]}__close"].to_numpy(dtype=float)
    cb_cfg = cfg.get("circuit_breaker", {}) or {}
    scale, reason = vol_scale(
        btc_closes,
        window=int(cb_cfg.get("window", 20)),
        threshold_high=float(cb_cfg.get("threshold_high", 0.06)),
        threshold_mid=float(cb_cfg.get("threshold_mid", 0.045)),
    )
    target = target * scale

    # Hard investment limits (size caps, per-trade floors/ceilings).
    current_holdings_usd = holdings * price
    target, limit_notes = apply_limits(target, current_holdings_usd, equity, cfg)

    result = broker.rebalance(target, price, ts=ts)
    result["target_weights"] = target.tolist()
    result["price"] = price.tolist()
    result["assets"] = assets
    result["circuit_breaker"] = {"scale": scale, "reason": reason}
    result["limits_applied"] = limit_notes
    result["threshold_skipped"] = threshold_skipped
    result["halted"] = False
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default="models/policy.zip")
    ap.add_argument("--mode", choices=["paper", "once"], default="once")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db_path = Path(ROOT) / "data" / "paper.db"
    ensure_dir(db_path.parent)
    assets = [a["symbol"].split("/")[0] for a in cfg["assets"]]
    broker = PaperBroker(
        db_path,
        assets=assets,
        initial_cash=cfg["initial_cash"],
        fee_bps=cfg["fee_bps"],
        slippage_std=[a["slippage_std"] for a in cfg["assets"]],
        borrow_rate_daily=float(cfg.get("borrow_rate_daily", 0.0)),
    )
    model_path = Path(ROOT) / args.model
    prefix = cfg.get("model_prefix", "policy_s")
    has_ensemble = bool(list((Path(ROOT) / "models").glob(f"{prefix}*.zip")))
    if not model_path.exists() and not has_ensemble:
        print(f"[live] no model found (neither {model_path} nor models/{prefix}*.zip) — train first")
        return

    result = run_once(cfg, model_path, broker)
    if result.get("error"):
        print(f"[live] run failed: {result['error']}")
        return
    print(f"[live] equity={result['equity']:.2f} weights={result['weights']} target={result.get('target_weights')}")


if __name__ == "__main__":
    main()
