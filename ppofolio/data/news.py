"""News + sentiment.

GNews API for headlines (requires GNEWS_API_KEY). Sentiment via OpenAI gpt-4o-mini
(requires OPENAI_API_KEY). Cached as one row per (date, asset) with:
  - sent_mean:  LLM score in [-1, 1]
  - sent_count: number of headlines used that day (0 when no key / no coverage)

All LLM responses are cached on disk by content hash (see sentiment_llm.py), so
re-runs are free. Without OPENAI_API_KEY the function returns zero sentiment
(and the pipeline still works — the feature just becomes uninformative).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from ppofolio.data.sentiment_llm import score_day
from ppofolio.utils import ROOT, ensure_dir, env


def fetch_gnews_day(query: str, day: pd.Timestamp, max_items: int = 30) -> list[dict]:
    key = env("GNEWS_API_KEY")
    if not key:
        return []
    frm = day.strftime("%Y-%m-%dT00:00:00Z")
    to = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    params = {"q": query, "lang": "en", "from": frm, "to": to, "max": max_items, "apikey": key}
    r = requests.get("https://gnews.io/api/v4/search", params=params, timeout=15)
    if r.status_code != 200:
        return []
    return r.json().get("articles", [])


def aggregate_day(query: str, asset_slug: str, day: pd.Timestamp, max_items: int) -> tuple[float, int]:
    items = fetch_gnews_day(query, day, max_items)
    if not items:
        return 0.0, 0
    headlines = [f"{it.get('title', '')}. {it.get('description', '')}".strip(". ") for it in items]
    result = score_day(asset_slug, day.strftime("%Y-%m-%d"), headlines)
    # Weight by confidence so low-confidence days don't push the policy around.
    weighted = float(result["score"]) * float(result["confidence"])
    return weighted, len(headlines)


def load_or_fetch(query: str, asset_slug: str, start: str, end: str, cache_dir: str, max_items: int = 30) -> pd.DataFrame:
    path = Path(ROOT) / cache_dir / f"{asset_slug}.parquet"
    ensure_dir(path.parent)
    idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")

    if path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index, utc=True)
        cached = cached.reindex(idx)
        need = cached[cached["sent_count"].isna()].index
    else:
        cached = pd.DataFrame(index=idx, columns=["sent_mean", "sent_count"], dtype=float)
        need = idx

    can_fetch = bool(env("GNEWS_API_KEY")) and bool(env("OPENAI_API_KEY"))
    if can_fetch and len(need) > 0:
        for d in need:
            try:
                m, c = aggregate_day(query, asset_slug, d, max_items)
            except Exception as e:
                print(f"[news] {query} {d.date()} failed: {e}")
                m, c = 0.0, 0
            cached.loc[d, ["sent_mean", "sent_count"]] = [m, c]
            time.sleep(1.1)  # GNews free-tier courtesy delay

    cached = cached.fillna({"sent_mean": 0.0, "sent_count": 0})
    cached.to_parquet(path)
    return cached.loc[idx]
