"""Market-wide macro features that apply to every asset row.

Three free sources, no API keys needed:
  - Fear & Greed Index:  https://api.alternative.me/fng/
  - BTC dominance:        CoinGecko /global (daily snapshot)
  - Funding-rate proxy:   Binance futures premium index (mark - spot) for BTCUSDT

Cached per-source. Returned joined to a daily index.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from ppofolio.utils import ROOT, ensure_dir


def fetch_fng(start: str, end: str) -> pd.DataFrame:
    """Fear & Greed Index (0-100). Scaled to [-1, 1] for the policy."""
    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s", utc=True).dt.normalize()
    df["fng"] = pd.to_numeric(df["value"], errors="coerce") / 50.0 - 1.0  # -> [-1, 1]
    return df.set_index("date")[["fng"]].sort_index()


def fetch_btc_dominance(start: str, end: str) -> pd.DataFrame:
    """BTC dominance from CoinGecko.

    Free CoinGecko endpoint only returns a snapshot, not history, without a key.
    For live use we record today's value and forward-fill; for history we fall back
    to a constant 50% (a reasonable neutral prior). This feature is small anyway.
    """
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=20)
        r.raise_for_status()
        dom = r.json()["data"]["market_cap_percentage"].get("btc", 50.0)
    except Exception:
        dom = 50.0
    idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    # We only have today's value — broadcast. It'll be uninformative over long history
    # but adds no noise either. Real value is for the live prediction.
    return pd.DataFrame({"btc_dominance": (dom - 50.0) / 50.0}, index=idx)


def fetch_funding_rate(start: str, end: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Daily-mean Binance perpetual futures funding rate for BTCUSDT.

    Funding rates are paid every 8h. High positive = longs pay shorts = bullish crowd.
    """
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows: list[dict] = []
    ts = start_ms
    while ts < end_ms:
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol, "startTime": ts, "limit": 1000},
                timeout=20,
            )
            r.raise_for_status()
            batch = r.json()
        except Exception:
            break
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["fundingTime"]
        if last_ts <= ts:
            break
        ts = last_ts + 1
        time.sleep(0.15)
    if not rows:
        idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
        return pd.DataFrame({"funding_rate": 0.0}, index=idx)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True).dt.normalize()
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    daily = df.groupby("date")["funding_rate"].mean().to_frame()
    # Scale: funding is typically 0.01% per 8h = 0.0001; multiply by 10k for ~[-1, 1]
    daily["funding_rate"] = (daily["funding_rate"] * 10_000).clip(-2.0, 2.0)
    return daily


def broadcast_to_index(daily_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a daily-indexed dataframe to any target index via forward-fill.

    Used to broadcast daily macro (Fear&Greed, BTC dominance, funding) to an
    hourly bar grid. Values hold constant within each day.
    """
    if daily_df.empty:
        return pd.DataFrame(0.0, index=target_index, columns=daily_df.columns)
    dd = daily_df.copy()
    dd.index = pd.to_datetime(dd.index, utc=True).normalize()
    # Pandas .reindex(method="ffill") fills new values forward from the previous known value.
    # We normalize target timestamps to their date so that each hourly ts gets that day's value.
    target_day = pd.to_datetime(target_index, utc=True).normalize()
    mapped = dd.reindex(target_day, method="ffill")
    mapped.index = target_index
    return mapped


def load_or_fetch_macro(start: str, end: str, cache_dir: str) -> pd.DataFrame:
    path = Path(ROOT) / cache_dir / "macro.parquet"
    ensure_dir(path.parent)
    if path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index, utc=True)
        req_start = pd.Timestamp(start, tz="UTC")
        req_end = pd.Timestamp(end, tz="UTC")
        if cached.index.min() <= req_start and cached.index.max() >= req_end - pd.Timedelta(days=1):
            return cached.loc[req_start: req_end - pd.Timedelta(days=1)]

    try:
        fng = fetch_fng(start, end)
    except Exception as e:
        print(f"[macro] fng fetch failed: {e}")
        fng = pd.DataFrame()
    try:
        dom = fetch_btc_dominance(start, end)
    except Exception as e:
        print(f"[macro] btc dom failed: {e}")
        dom = pd.DataFrame()
    try:
        fr = fetch_funding_rate(start, end)
    except Exception as e:
        print(f"[macro] funding rate failed: {e}")
        fr = pd.DataFrame()

    idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    df = pd.DataFrame(index=idx)
    for part in (fng, dom, fr):
        if part is not None and not part.empty:
            df = df.join(part, how="left")
    for c in ("fng", "btc_dominance", "funding_rate"):
        if c not in df.columns:
            df[c] = 0.0
    df = df[["fng", "btc_dominance", "funding_rate"]].ffill(limit=7).fillna(0.0)
    df.to_parquet(path)
    return df
