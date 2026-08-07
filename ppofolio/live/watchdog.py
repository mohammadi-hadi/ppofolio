"""Intraday risk watchdog.

Runs every hour, does NOT trigger a rebalance. Only checks whether the
kill-switch should engage based on intraday equity change. If engaged,
forces cash + alert; otherwise just logs and returns.

Rationale: the model decides daily (we can't beat that without retraining).
But intraday flash-crashes deserve protection. This gives us it cheaply.
"""
from __future__ import annotations

import logging

import numpy as np

from ppofolio.data import ohlcv
from ppofolio.live.kill_switch import check_and_update, is_halted
from ppofolio.utils import load_config

log = logging.getLogger(__name__)


def run_watchdog(cfg_path: str, broker) -> dict:
    cfg = load_config(cfg_path)
    assets = [a["symbol"] for a in cfg["assets"]]
    prices: list[float] = []
    import pandas as pd
    # Hourly closed bars: at most 1h stale, and they never pollute the daily
    # cache with a partial candle (the old code fetched daily bars incl. today's).
    end = pd.Timestamp.now(tz="UTC").floor("1h")
    start = end - pd.Timedelta(hours=72)
    try:
        for sym in assets:
            df = ohlcv.load_or_fetch(sym, start.isoformat(), end.isoformat(),
                                     cfg["data_paths"]["ohlcv"], timeframe="1h")
            prices.append(float(df["close"].iloc[-1]))
    except Exception as e:
        log.warning("watchdog price fetch failed: %s", e)
        return {"ok": False, "error": str(e)}

    price_arr = np.asarray(prices, dtype=float)
    equity = broker.equity(price_arr)
    halt_now, reason = check_and_update(broker, equity, cfg)
    if halt_now or is_halted(broker):
        # Force a cash rebalance to lock in — but only while there is actually
        # exposure. Rebalancing an already-flat book every hour produced an
        # endless stream of float-precision dust trades.
        gross = float(np.abs(broker.holdings() * price_arr).sum())
        if gross > 1.0:
            ts = int(pd.Timestamp.now(tz="UTC").timestamp())
            broker.rebalance(np.zeros(len(assets)), price_arr, ts=ts)
            log.warning("watchdog halted trading, liquidated to cash: %s", reason)
        else:
            log.info("watchdog: halted and flat (gross=%.2f), nothing to do", gross)
    return {"ok": True, "halted": is_halted(broker), "equity": equity, "reason": reason}
