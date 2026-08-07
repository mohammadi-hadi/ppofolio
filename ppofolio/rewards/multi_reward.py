"""Multi-objective reward (paper Meta-RL-Crypto §3.2).

Each component is normalized to roughly [-1, 1] per-step so linear weights are comparable.
Components:
- R_return:   per-bar net P&L as a log return (matches ppofolio/backtest/metrics.py).
- R_sharpe:   incremental Sharpe over a rolling window (EWMA-ish).
- R_dd:       -max drawdown from running peak, scaled.
- R_liquidity: penalty when traded notional > threshold * daily volume.
- R_sentiment: cosine sim between sign(weights) and sign(sentiment) vectors.
- R_loss_asymm: quadratic penalty on negative returns only.
- R_borrow:   per-bar borrow cost on short exposure.
- R_turnover: actually-traded notional as a fraction of equity — holding costs 0.

Timeframe scaling: terms whose per-bar |magnitude| does not shrink with the bar
size (drawdown and sentiment are levels; the sharpe DELTA is tanh-bounded noise
whose absolute mass does not telescope) are divided by bars_per_day so their
per-DAY mass is constant across timeframes — verified empirically by
scripts/reward_audit.py. Return/liquidity/borrow/turnover are flows and
loss_asymm is variance-additive (24 × E[(r_day/√24)²] = E[r_day²]); those are
left alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class RewardState:
    returns: deque = field(default_factory=lambda: deque(maxlen=60))
    peak_equity: float = 1.0

    @classmethod
    def create(cls, peak_equity: float = 1.0, sharpe_window: int = 60) -> "RewardState":
        return cls(returns=deque(maxlen=int(sharpe_window)), peak_equity=peak_equity)


def _safe_sharpe(returns: list[float], bars_per_year: int = 365) -> float:
    if len(returns) < 5:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    sd = arr.std()
    if sd < 1e-8:
        return 0.0
    return float(np.sqrt(bars_per_year) * arr.mean() / sd)


def compute(
    *,
    prev_equity: float,
    new_equity: float,
    weights: np.ndarray,
    prev_weights: np.ndarray,
    traded_notional: np.ndarray,
    daily_volume_usd: np.ndarray,
    sentiment: np.ndarray,
    state: RewardState,
    cfg: dict,
) -> tuple[float, dict]:
    """Return (scalar_reward, component_dict)."""
    bars_per_day = max(int(cfg.get("bars_per_day", 1)), 1)

    # R_return — log return, the same quantity metrics.from_equity reports.
    r_ret = float(np.log(max(new_equity, 1e-9) / max(prev_equity, 1e-9)))

    # Sharpe contribution: change in rolling Sharpe from adding this bar's return
    bars_per_year = int(cfg.get("bars_per_year", 365))
    prev_sharpe = _safe_sharpe(list(state.returns), bars_per_year)
    state.returns.append(r_ret)
    new_sharpe = _safe_sharpe(list(state.returns), bars_per_year)
    r_sharpe = np.tanh(new_sharpe - prev_sharpe)

    # Drawdown
    state.peak_equity = max(state.peak_equity, new_equity)
    dd = (new_equity - state.peak_equity) / max(state.peak_equity, 1e-9)  # ≤ 0
    r_dd = dd  # already negative; bigger drawdown = more negative

    # Liquidity: penalize notional above threshold fraction of daily volume
    thr = cfg["liquidity_threshold_pct"]
    frac = np.where(daily_volume_usd > 0, traded_notional / daily_volume_usd, 0.0)
    r_liq = -float(np.clip((frac - thr).clip(min=0).sum(), 0.0, 1.0))

    # Sentiment alignment: cosine sim of sign(weights) and sign(sentiment)
    w_sign = np.sign(weights)
    s_sign = np.sign(sentiment)
    if np.any(w_sign) and np.any(s_sign):
        num = float((w_sign * s_sign).sum())
        den = float(np.linalg.norm(w_sign) * np.linalg.norm(s_sign) + 1e-9)
        r_sent = num / den
    else:
        r_sent = 0.0

    # Asymmetric loss penalty: only fires on negative returns, quadratic,
    # so a 10% loss is 100x worse than a 1% loss (instead of 10x).
    r_loss_asymm = -float((min(0.0, r_ret)) ** 2 * 100.0)

    # Borrow cost for shorts — scaled per-bar. weights here are post-trade marked
    # to market; negative component × borrow_rate_per_bar.
    borrow_per_bar = float(cfg.get("borrow_rate_per_bar", 0.0))
    short_notional_frac = float(np.sum(np.maximum(0.0, -weights)))
    r_borrow = -borrow_per_bar * short_notional_frac

    # Turnover penalty: what was actually TRADED this bar, as a fraction of equity.
    # Holding a position (even a volatile one) costs exactly 0 — the old
    # |weights - prev_weights| formulation punished passive price drift and drove
    # the v6.1 hourly policy into pathological flatness.
    r_turnover = -float(np.sum(traded_notional)) / max(prev_equity, 1e-9)

    w = cfg["weights"]
    scalar = (
        w["return"] * r_ret
        + (w["sharpe"] / bars_per_day) * r_sharpe
        + (w["drawdown"] / bars_per_day) * r_dd
        + w["liquidity"] * r_liq
        + (w["sentiment"] / bars_per_day) * r_sent
        + w.get("loss_asymm", 0.0) * r_loss_asymm
        + w.get("borrow", 0.0) * r_borrow
        + w.get("turnover", 0.0) * r_turnover
    )

    return float(scalar), {
        "r_return": r_ret,
        "r_sharpe": float(r_sharpe),
        "r_drawdown": float(r_dd),
        "r_liquidity": r_liq,
        "r_sentiment": r_sent,
        "r_loss_asymm": r_loss_asymm,
        "r_borrow": r_borrow,
        "r_turnover": r_turnover,
    }
