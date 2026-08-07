"""Per-asset technical indicators computed from OHLCV."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands


def _regime_features(daily_close: pd.Series) -> pd.DataFrame:
    """Calendar-day regime features from a series of DAILY closes."""
    out = pd.DataFrame(index=daily_close.index)
    out["ret_60d"] = daily_close.pct_change(60).clip(-1.0, 3.0)
    out["ret_180d"] = daily_close.pct_change(180).clip(-1.0, 3.0)
    log_ret = np.log(daily_close / daily_close.shift(1))
    vol20 = log_ret.rolling(20).std()
    vol_mu = vol20.rolling(200, min_periods=30).mean()
    vol_sd = vol20.rolling(200, min_periods=30).std().replace(0, np.nan)
    out["vol_20d"] = ((vol20 - vol_mu) / vol_sd).clip(-5, 5)
    ma200 = daily_close.rolling(200, min_periods=50).mean()
    out["ma200_dist"] = ((daily_close - ma200) / ma200).clip(-1, 3)
    return out


def add_technical(df: pd.DataFrame, bars_per_day: int = 1,
                  daily_close: pd.Series | None = None) -> pd.DataFrame:
    """Input: OHLCV dataframe. Output: original + indicator columns.

    Short-window oscillators (rsi_14, macd, bb_pct, atr_14, log_vol_7d, ret_1d)
    are BAR-indexed by design — at hourly they read the last 14/26/20 hours.
    The regime features (ret_60d, ret_180d, vol_20d, ma200_dist) are CALENDAR
    days at every timeframe: at intraday timeframes they are computed on daily
    closes (resampled from df, or passed in via daily_close for live panels that
    are too short to resample) and forward-filled onto the bar grid.
    """
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change().clip(-0.5, 0.5)
    # z-score log-volume so the feature sits in ~[-3, 3]
    lv = np.log(out["volume"].rolling(7).mean().replace(0, np.nan))
    out["log_vol_7d"] = ((lv - lv.rolling(90, min_periods=20).mean()) /
                         lv.rolling(90, min_periods=20).std().replace(0, np.nan))
    out["rsi_14"] = RSIIndicator(out["close"], window=14).rsi() / 100.0
    macd = MACD(out["close"])
    out["macd"] = macd.macd_diff() / out["close"]
    bb = BollingerBands(out["close"])
    denom = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
    out["bb_pct"] = (out["close"] - bb.bollinger_lband()) / denom
    out["atr_14"] = AverageTrueRange(out["high"], out["low"], out["close"], window=14).average_true_range() / out["close"]

    # Regime features — give the policy a longer memory than the lookback window.
    if bars_per_day <= 1:
        reg = _regime_features(out["close"])
        for c in reg.columns:
            out[c] = reg[c]
    else:
        if daily_close is None:
            daily_close = out["close"].resample("1D").last().dropna()
        reg = _regime_features(daily_close.sort_index())
        # A day's close is only known once the day is over: stamp each daily value
        # at the NEXT day's open before ffilling onto the bar grid (no look-ahead).
        reg.index = reg.index + pd.Timedelta(days=1)
        reg = reg.reindex(out.index, method="ffill")
        for c in reg.columns:
            out[c] = reg[c]
    return out
