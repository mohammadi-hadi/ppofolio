"""Hard limits applied on top of the policy's proposed weights.

These run AFTER the circuit-breaker and BEFORE the broker executes:
  1. max_single_position: no more than X fraction of equity in one asset
  2. max_gross_exposure:  total |weights| capped (already clipped by action space but re-enforced)
  3. max_portfolio_usd:   if equity > this, scale all weights so we don't deploy >max
  4. max_trade_notional:  reject a rebalance that would move more than $X on one asset
  5. min_trade_notional:  ignore trades below this (avoid exchange minimums — set to 10 for real exchanges)
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def apply_limits(
    target_w: np.ndarray,
    current_holdings_usd: np.ndarray,
    equity: float,
    cfg: dict,
) -> tuple[np.ndarray, list[str]]:
    """Return (clipped_weights, notes). `notes` is human-readable log of what was changed."""
    limits = cfg.get("invest_limits", {}) or {}
    notes: list[str] = []
    out = np.asarray(target_w, dtype=float).copy()

    max_single = float(limits.get("max_single_position", 1.0))
    max_short = float(limits.get("max_single_short", max_single))
    if max_single < 1.0 or max_short < 1.0:
        before = out.copy()
        # asymmetric caps: longs capped at max_single, shorts capped at max_short
        out = np.where(out >= 0, np.minimum(out, max_single), np.maximum(out, -max_short))
        if not np.array_equal(before, out):
            notes.append(f"capped long={max_single:.0%} short={max_short:.0%}")

    max_gross = float(limits.get("max_gross_exposure", 1.0))
    gross = float(np.abs(out).sum())
    if gross > max_gross and gross > 0:
        out = out * (max_gross / gross)
        notes.append(f"scaled gross exposure {gross:.2f}->{max_gross:.2f}")

    cap = limits.get("max_portfolio_usd")
    if cap is not None and equity > float(cap):
        scale = float(cap) / equity
        out = out * scale
        notes.append(f"equity {equity:.0f} > cap {cap}, scaled weights x{scale:.3f}")

    min_notional = float(limits.get("min_trade_notional", 0.0))
    max_notional = float(limits.get("max_trade_notional", float("inf")))
    target_notional = out * equity
    delta = target_notional - current_holdings_usd
    # drop tiny trades
    tiny = np.abs(delta) < min_notional
    if min_notional > 0 and tiny.any():
        # revert to current weights for tiny-delta assets
        for i in np.where(tiny)[0]:
            if equity > 0:
                out[i] = current_holdings_usd[i] / equity
        notes.append(f"dropped {int(tiny.sum())} trades below ${min_notional:g}")
    # cap big trades
    too_big = np.abs(delta) > max_notional
    if np.isfinite(max_notional) and too_big.any():
        for i in np.where(too_big)[0]:
            capped = np.sign(delta[i]) * max_notional
            new_target = current_holdings_usd[i] + capped
            out[i] = new_target / equity if equity > 0 else 0.0
        notes.append(f"capped {int(too_big.sum())} trades at ${max_notional:g}")

    return out, notes
