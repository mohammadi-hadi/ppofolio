"""Backtest metrics matching paper §4.1."""
from __future__ import annotations

import numpy as np
import pandas as pd


def from_equity(equity: pd.Series, bars_per_year: int = 365) -> dict:
    e = equity.dropna().astype(float)
    if len(e) < 2:
        return {"total_return": 0.0, "daily_mean": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    log_ret = np.log(e / e.shift(1)).dropna()
    total_return = float(e.iloc[-1] / e.iloc[0] - 1.0)
    daily_mean = float(log_ret.mean())
    sharpe = float(np.sqrt(bars_per_year) * log_ret.mean() / (log_ret.std() + 1e-12))
    peak = e.cummax()
    max_dd = float(((e - peak) / peak).min())
    win_rate = float((log_ret > 0).mean())
    return {
        "total_return": total_return,
        "daily_mean": daily_mean,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }
