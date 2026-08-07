"""Assemble a single multi-asset feature panel indexed by date.

Output columns follow: "{asset}__{feature}" for per-asset features.
We return a dict of per-asset frames (long format) AND a wide frame for the env.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(s: pd.Series, win: int = 90) -> pd.Series:
    mu = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std().replace(0, np.nan)
    return (s - mu) / sd


def build_panel(per_asset: dict[str, pd.DataFrame], tech_cols: list[str], onchain_cols: list[str], sent_cols: list[str]) -> pd.DataFrame:
    """Concat wide: columns like BTC__ret_1d, BTC__close, ETH__ret_1d, etc.

    Also keeps raw close + volume per asset (needed by the env for P&L).
    """
    out = []
    for asset, df in per_asset.items():
        keep = df[["close", "volume"] + tech_cols + onchain_cols + sent_cols].copy()
        # z-score on-chain (scale-free across assets)
        for c in onchain_cols:
            if c in keep:
                keep[c] = zscore(keep[c])
        keep.columns = [f"{asset}__{c}" for c in keep.columns]
        out.append(keep)
    panel = pd.concat(out, axis=1).sort_index()
    return panel


def finalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill on-chain/sentiment gaps, drop rows with missing prices."""
    panel = panel.ffill(limit=5)
    close_cols = [c for c in panel.columns if c.endswith("__close")]
    panel = panel.dropna(subset=close_cols)
    panel = panel.fillna(0.0)
    return panel
