"""Sentiment scoring via gpt-4o-mini (or any OpenAI chat model).

Input: a list of headlines/snippets for one (asset, day).
Output: a dict with `score` in [-1, 1] and `confidence` in [0, 1].

Batched in one call per day-asset to keep costs low:
  - 30 headlines × 3 assets × 365 days ≈ 33k days; one request each ≈ $1–3 at
    gpt-4o-mini pricing. Responses are cached to disk by content hash so repeats
    are free.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI

from ppofolio.utils import ROOT, ensure_dir

_CACHE_DIR = ensure_dir(Path(ROOT) / "data" / "llm_cache")
_MODEL = os.environ.get("OPENAI_SENTIMENT_MODEL", "gpt-4o-mini")
_SYSTEM = (
    "You are a financial-sentiment rater for cryptocurrency news. "
    "Given headlines for a single asset on a single day, return strict JSON "
    '{"score": float in [-1, 1], "confidence": float in [0, 1], "rationale": "one sentence"}. '
    "score = -1 strongly bearish, 0 neutral/irrelevant, +1 strongly bullish. "
    "Weight credibility and specificity over hype. Return ONLY the JSON object."
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _cache_key(asset: str, day: str, items: list[str]) -> Path:
    h = hashlib.sha256(("|".join([_MODEL, asset, day] + items)).encode()).hexdigest()[:24]
    return _CACHE_DIR / f"{h}.json"


def score_day(asset: str, day: str, headlines: list[str]) -> dict:
    """Return {"score", "confidence", "rationale"} for one asset-day.

    Always returns a dict; on any failure returns score=0, confidence=0.
    """
    headlines = [h for h in headlines if h]
    if not headlines:
        return {"score": 0.0, "confidence": 0.0, "rationale": "no headlines"}

    cache = _cache_key(asset, day, headlines)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    try:
        client = _get_client()
        user = f"Asset: {asset}\nDate: {day}\nHeadlines:\n- " + "\n- ".join(headlines[:30])
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        out = {
            "score": float(max(-1.0, min(1.0, data.get("score", 0.0)))),
            "confidence": float(max(0.0, min(1.0, data.get("confidence", 0.0)))),
            "rationale": str(data.get("rationale", ""))[:200],
        }
        cache.write_text(json.dumps(out))
        return out
    except Exception as e:
        print(f"[sentiment_llm] {asset} {day} failed: {e}")
        return {"score": 0.0, "confidence": 0.0, "rationale": f"error: {e}"[:200]}
