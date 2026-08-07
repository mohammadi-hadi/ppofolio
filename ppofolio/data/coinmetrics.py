"""CoinMetrics Community API (no key needed) for on-chain metrics.

Docs: https://docs.coinmetrics.io/api/v4
Community tier gives daily granularity for a curated set of metrics.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from ppofolio.utils import ROOT, ensure_dir

BASE = "https://community-api.coinmetrics.io/v4"
METRICS = ["AdrActCnt", "TxCnt", "FeeMeanUSD"]
COL_MAP = {"AdrActCnt": "active_addresses", "TxCnt": "tx_count", "FeeMeanUSD": "mean_fee_usd"}
# The community API rejects the default python-requests User-Agent with 403.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
RETRIES = 3


def fetch_onchain(asset: str, start: str, end: str) -> pd.DataFrame:
    metrics = list(METRICS)
    params = {
        "assets": asset,
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start,
        "end_time": end,
        "page_size": 10_000,
    }
    api_key = os.environ.get("COINMETRICS_API_KEY")
    if api_key:
        params["api_key"] = api_key
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(f"{BASE}/timeseries/asset-metrics", params=params,
                             headers=HEADERS, timeout=30)
            if r.status_code == 403:
                # The community catalog shrinks over time (FeeMeanUSD was dropped
                # 2026-07): a 403 naming a metric means "not in your tier", not a
                # rate ban. Drop the named metric and retry with the rest.
                msg = r.json().get("error", {}).get("message", "") if r.content else ""
                dropped = [m for m in metrics if m in msg and "not available" in msg]
                if dropped:
                    metrics = [m for m in metrics if m not in dropped]
                    if not metrics:
                        r.raise_for_status()
                    print(f"[coinmetrics] {asset}: {','.join(dropped)} not in community tier, "
                          f"continuing with {','.join(metrics)}")
                    params["metrics"] = ",".join(metrics)
                    continue
            r.raise_for_status()
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Community tier rate-limits hard; give it room before retrying.
            time.sleep(5 * (attempt + 1) ** 2)
    if last_err is not None:
        raise last_err
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame(columns=list(COL_MAP.values())).assign(date=pd.NaT).set_index("date")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    df = df.rename(columns=COL_MAP)
    for c in COL_MAP.values():
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:  # metric unavailable in this tier — keep the column, downstream fills
            df[c] = float("nan")
    return df.set_index("date")[list(COL_MAP.values())].sort_index()


def load_or_fetch(asset: str, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    path = Path(ROOT) / cache_dir / f"{asset}.parquet"
    ensure_dir(path.parent)
    cached = None
    if path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index, utc=True)
        if cached.index.min() <= pd.Timestamp(start, tz="UTC") and cached.index.max() >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=1):
            return cached
    try:
        df = fetch_onchain(asset, start, end)
    except Exception as e:
        print(f"[coinmetrics] {asset} fetch failed after {RETRIES} attempts: {e}.")
        if cached is not None:
            # Serve stale history instead of nothing — downstream ffill/zero-fill
            # handles the gap, and the cache is never poisoned with the fallback.
            print(f"[coinmetrics] serving stale cache through {cached.index.max().date()}")
            return cached
        idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
        return pd.DataFrame(index=idx, columns=list(COL_MAP.values())).astype(float)
    if cached is not None:
        # Merge instead of clobber: a narrow live request must not shrink the
        # multi-year training cache.
        df = pd.concat([cached, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    return df
