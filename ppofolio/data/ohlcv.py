"""Fetch + cache OHLCV bars from Binance via ccxt, parameterized by timeframe."""
from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

from ppofolio.utils import ROOT, ensure_dir

MS_PER_UNIT = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
               "4h": 14_400_000, "1d": 86_400_000}


def _client(exchange: str = "binance"):
    cls = getattr(ccxt, exchange)
    return cls({"enableRateLimit": True})


def _bar_ms(timeframe: str) -> int:
    if timeframe not in MS_PER_UNIT:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return MS_PER_UNIT[timeframe]


def fetch_ohlcv(symbol: str, start: str, end: str, exchange: str = "binance",
                timeframe: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV in [start, end) as a dataframe indexed by UTC timestamp."""
    client = _client(exchange)
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    bar_ms = _bar_ms(timeframe)

    rows: list[list] = []
    while since < end_ms:
        batch = client.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= since:
            break
        since = last + bar_ms
        time.sleep(client.rateLimit / 1000.0)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    # For daily we normalize to date; for intraday keep the exact hour.
    stamps = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["date"] = stamps.dt.normalize() if timeframe == "1d" else stamps
    df = df[(df["date"] >= pd.Timestamp(start, tz="UTC")) & (df["date"] < pd.Timestamp(end, tz="UTC"))]
    # Only closed bars: drop the exchange's still-forming candle so callers (and the
    # cache) never see partial data. A bar stamped t closes at t + bar.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["date"] + pd.Timedelta(milliseconds=bar_ms) <= now]
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def load_or_fetch(symbol: str, start: str, end: str, cache_dir: str,
                  timeframe: str = "1d") -> pd.DataFrame:
    slug = symbol.replace("/", "-")
    # cache per timeframe so daily and hourly don't collide
    suffix = "" if timeframe == "1d" else f"-{timeframe}"
    path = Path(ROOT) / cache_dir / f"{slug}{suffix}.parquet"
    ensure_dir(path.parent)
    bar = pd.Timedelta(milliseconds=_bar_ms(timeframe))
    req_start = pd.Timestamp(start, tz="UTC")
    req_end = pd.Timestamp(end, tz="UTC")
    cached = None
    if path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index, utc=True)
        have_start = cached.index.min()
        # for daily, the last cached row represents 1 calendar day; for intraday, 1 bar
        have_end = cached.index.max() + bar
        if have_start <= req_start and have_end >= req_end:
            return cached.loc[req_start:req_end - bar]
        if have_start <= req_start and have_end < req_end:
            # incremental: fetch only the missing tail instead of the full range.
            # Start one bar early so a stale partial final row (written by older
            # code) is refetched and replaced by its closed version on merge.
            df = fetch_ohlcv(symbol, (have_end - bar).isoformat(), end, timeframe=timeframe)
        else:
            df = fetch_ohlcv(symbol, start, end, timeframe=timeframe)
    else:
        df = fetch_ohlcv(symbol, start, end, timeframe=timeframe)
    if cached is not None:
        df = pd.concat([cached, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    return df.loc[req_start:req_end - bar]
