"""Daily closes from Yahoo Finance's public chart API (for assets Binance lacks,
e.g. COMEX silver futures SI=F). One request per fetch, parquet-cached; the
cache is only written on success so a bad response can't poison it."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from ppofolio.utils import ROOT, ensure_dir

BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
HEADERS = {"User-Agent": "Mozilla/5.0"}
RETRIES = 3


def fetch_daily_closes(symbol: str, range_: str = "3y") -> pd.Series:
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(f"{BASE}/{symbol}", params={"interval": "1d", "range": range_},
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()["chart"]["result"][0]
            ts = data["timestamp"]
            closes = data["indicators"]["quote"][0]["close"]
            idx = pd.to_datetime(ts, unit="s", utc=True).normalize()
            s = pd.Series(closes, index=idx, name="close", dtype=float).dropna()
            # Last row can be a live (unfinished) session — drop anything from today.
            today = pd.Timestamp.now(tz="UTC").normalize()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            return s[s.index < today]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


def load_or_fetch_closes(symbol: str, cache_dir: str, max_age_hours: float = 12.0) -> pd.Series:
    slug = symbol.replace("=", "-").replace("/", "-").lower()
    path = Path(ROOT) / cache_dir / f"yahoo-{slug}.parquet"
    ensure_dir(path.parent)
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        df = pd.read_parquet(path)
        s = df["close"]
        s.index = pd.to_datetime(s.index, utc=True)
        return s
    try:
        s = fetch_daily_closes(symbol)
    except Exception as e:
        if path.exists():
            print(f"[yahoo] {symbol} fetch failed: {e}. Serving stale cache.")
            df = pd.read_parquet(path)
            s = df["close"]
            s.index = pd.to_datetime(s.index, utc=True)
            return s
        raise
    s.to_frame().to_parquet(path)
    return s
