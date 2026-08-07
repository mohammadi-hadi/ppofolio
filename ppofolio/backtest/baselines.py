"""Simple baselines: buy-and-hold (with and without rebalancing) and MACD crossover.

All baselines charge the same fee_bps on traded notional as the PPO simulation so
the comparison is fair. Slippage is left out: the sim's slippage is zero-mean
gaussian, so it has no expected cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_from_weights(prices: pd.DataFrame, weights: pd.DataFrame,
                        initial_cash: float = 1_000_000, fee_bps: float = 0.0) -> pd.Series:
    """Simulate equity for a target-weight strategy, charging fee_bps on turnover.

    Turnover at each step is measured against the DRIFTED weights (what the
    portfolio drifted to), not yesterday's targets — the same notion of traded
    notional the env and brokers use.
    """
    prices = prices.astype(float)
    weights = weights.reindex(prices.index).fillna(0.0)
    ret = prices.pct_change().fillna(0.0).to_numpy()
    tgt = weights.to_numpy()
    fee = fee_bps / 10_000.0

    n = prices.shape[1]
    eq = float(initial_cash)
    w_prev = np.zeros(n)
    out = np.empty(len(prices))
    for t in range(len(prices)):
        r = ret[t]
        port_ret = float((w_prev * r).sum())
        eq *= (1.0 + port_ret)
        denom = 1.0 + port_ret
        w_drift = w_prev * (1.0 + r) / denom if abs(denom) > 1e-12 else w_prev
        turnover = float(np.abs(tgt[t] - w_drift).sum())
        eq -= eq * fee * turnover
        w_prev = tgt[t]
        out[t] = eq
    return pd.Series(out, index=prices.index, name="equity")


def buy_and_hold(prices: pd.DataFrame, initial_cash: float = 1_000_000,
                 fee_bps: float = 0.0) -> pd.Series:
    """TRUE buy-and-hold: buy equal notional at t0 (paying the fee once), never trade
    again. Weights drift with prices."""
    prices = prices.astype(float)
    fee = fee_bps / 10_000.0
    alloc = initial_cash / prices.shape[1]
    units = alloc * (1.0 - fee) / prices.iloc[0]
    eq = (prices * units).sum(axis=1)
    eq.name = "equity"
    return eq


def bh_rebalanced(prices: pd.DataFrame, initial_cash: float = 1_000_000,
                  fee_bps: float = 0.0) -> pd.Series:
    """Equal-weight portfolio rebalanced every bar (the old 'buy_and_hold')."""
    n = prices.shape[1]
    w = pd.DataFrame(np.full_like(prices, 1.0 / n, dtype=float), index=prices.index, columns=prices.columns)
    return equity_from_weights(prices, w, initial_cash, fee_bps=fee_bps)


def risk_parity(prices: pd.DataFrame, initial_cash: float = 1_000_000,
                fee_bps: float = 0.0, history: pd.DataFrame | None = None,
                vol_window: int = 30) -> pd.Series:
    """Inverse-volatility weights, fully invested, recomputed every bar.

    `history` (a superset of `prices` ending at the same bars) supplies warmup
    for the vol window so the strategy is invested from the first window bar.
    """
    ref = (history if history is not None else prices).astype(float)
    vol = ref.pct_change().rolling(vol_window, min_periods=max(vol_window // 3, 5)).std()
    inv = 1.0 / vol.where(vol > 0)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
    return equity_from_weights(prices, w.reindex(prices.index), initial_cash, fee_bps=fee_bps)


def tsmom_ma200(prices: pd.DataFrame, initial_cash: float = 1_000_000,
                fee_bps: float = 0.0, history: pd.DataFrame | None = None,
                ma_window: int = 200) -> pd.Series:
    """Time-series momentum: hold 1/n in each asset trading above its `ma_window`
    moving average, cash otherwise. The classic trend filter — a regime-aware
    bar the policy should clear in BOTH bull and bear windows."""
    ref = (history if history is not None else prices).astype(float)
    ma = ref.rolling(ma_window, min_periods=int(ma_window * 0.75)).mean()
    long = (ref > ma).astype(float) / ref.shape[1]
    return equity_from_weights(prices, long.reindex(prices.index), initial_cash, fee_bps=fee_bps)


def macd_crossover(prices: pd.DataFrame, initial_cash: float = 1_000_000,
                   fee_bps: float = 0.0) -> pd.Series:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    long = (macd > signal).astype(float)
    w = long.div(long.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # equity_from_weights already applies targets with a one-bar delay
    # (tgt[t] earns bar t+1 returns), so no extra shift here.
    return equity_from_weights(prices, w, initial_cash, fee_bps=fee_bps)
