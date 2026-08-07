"""Volatility circuit-breaker for the live loop.

Scales down total portfolio exposure when BTC realized vol exceeds thresholds.
Runs between model.predict() and broker.rebalance(). Doesn't touch training.

Rationale: if the market is moving harder than anything in the training distribution,
trust the model less. Go partly-cash unconditionally.
"""
from __future__ import annotations

import numpy as np


def vol_scale(btc_closes: np.ndarray, window: int = 20,
              threshold_high: float = 0.06, threshold_mid: float = 0.045) -> tuple[float, str]:
    """Can be called with kwargs from config so both daily and hourly agree."""
    """Return (scale_factor in [0,1], reason).

    - vol_20d > threshold_high  -> 0.25 (emergency)
    - vol_20d > threshold_mid   -> 0.50 (caution)
    - else                      -> 1.00 (normal)
    """
    if btc_closes is None or len(btc_closes) < window + 1:
        return 1.0, "insufficient-history"
    log_ret = np.log(btc_closes[1:] / np.maximum(btc_closes[:-1], 1e-9))
    vol = float(log_ret[-window:].std())
    if vol > threshold_high:
        return 0.25, f"EMERGENCY vol_20d={vol:.4f}"
    if vol > threshold_mid:
        return 0.50, f"caution vol_20d={vol:.4f}"
    return 1.0, f"normal vol_20d={vol:.4f}"
