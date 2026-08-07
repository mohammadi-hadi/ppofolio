"""Multi-asset price watcher — transparent rule-based buy/sell signals.

Deliberately NOT a learned model: the signals are classic, auditable rules
(200-day trend + MACD momentum + RSI stretch). The service checks each watched
asset once a day and sends a Telegram alert only when an asset's signal
CHANGES. Current state for all assets is at GET /signals (gold also at /gold).

States per asset:
  bullish  — price above its 200-day average AND momentum (MACD) positive
  bearish  — price below its 200-day average AND momentum negative
  neutral  — mixed (trend and momentum disagree)

RSI adds a stretch note: >70 overbought (rally stretched), <30 oversold
(sell-off stretched). Informational, not trade instructions.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ppofolio.data import ohlcv, yahoo
from ppofolio.live.alerts import send as send_alert
from ppofolio.live.kill_switch import _get_meta, _set_meta

log = logging.getLogger(__name__)

DEFAULT_ASSETS = [
    {"name": "Gold", "symbol": "PAXG/USDT", "source": "binance"},
    {"name": "Silver", "symbol": "SI=F", "source": "yahoo"},
    {"name": "Bitcoin", "symbol": "BTC/USDT", "source": "binance"},
]


def compute_signals(close: pd.Series) -> dict:
    """All rules on a series of DAILY closes. Needs ~220 bars of history."""
    ma200 = close.rolling(200, min_periods=150).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_diff = macd - macd_signal

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    price = float(close.iloc[-1])
    above_ma = bool(price > float(ma200.iloc[-1])) if not np.isnan(ma200.iloc[-1]) else True
    momentum_up = bool(macd_diff.iloc[-1] > 0)
    r = float(rsi.iloc[-1])

    if above_ma and momentum_up:
        state = "bullish"
    elif not above_ma and not momentum_up:
        state = "bearish"
    else:
        state = "neutral"
    rsi_zone = "overbought" if r > 70 else "oversold" if r < 30 else "normal"

    return {
        "price": round(price, 2),
        "ma200": round(float(ma200.iloc[-1]), 2) if not np.isnan(ma200.iloc[-1]) else None,
        "above_ma200": above_ma,
        "macd_diff": round(float(macd_diff.iloc[-1]), 4),
        "momentum_up": momentum_up,
        "rsi": round(r, 1),
        "rsi_zone": rsi_zone,
        "state": state,
        "as_of": str(close.index[-1].date()),
    }


_HISTORY_SCHEMA_PG = (
    "CREATE TABLE IF NOT EXISTS signal_history ("
    "id BIGSERIAL PRIMARY KEY, ts BIGINT NOT NULL, asset TEXT NOT NULL, "
    "prev_state TEXT, state TEXT NOT NULL, rsi_zone TEXT, "
    "price DOUBLE PRECISION, rsi DOUBLE PRECISION)"
)
_HISTORY_SCHEMA_SQLITE = (
    "CREATE TABLE IF NOT EXISTS signal_history ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, asset TEXT NOT NULL, "
    "prev_state TEXT, state TEXT NOT NULL, rsi_zone TEXT, price REAL, rsi REAL)"
)


def record_history(broker, ts: int, name: str, prev_state: str | None, sig: dict) -> None:
    """Append a signal state change so alert hit-rates can be scored later.
    Works with both PostgresBroker and the SQLite PaperBroker; never raises."""
    row = (int(ts), name, prev_state, sig["state"], sig["rsi_zone"], sig["price"], sig["rsi"])
    try:
        if hasattr(broker, "_conn"):
            with broker._conn() as c, c.cursor() as cur:
                cur.execute(_HISTORY_SCHEMA_PG)
                cur.execute(
                    "INSERT INTO signal_history(ts, asset, prev_state, state, rsi_zone, price, rsi) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)", row)
            return
        if hasattr(broker, "db"):
            with broker._lock:
                broker.db.execute(_HISTORY_SCHEMA_SQLITE)
                broker.db.execute(
                    "INSERT INTO signal_history(ts, asset, prev_state, state, rsi_zone, price, rsi) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", row)
                broker.db.commit()
    except Exception:
        log.exception("signal history write failed for %s", name)


def read_history(broker, limit: int = 100) -> list[dict]:
    cols = ("ts", "asset", "prev_state", "state", "rsi_zone", "price", "rsi")
    q = ("SELECT ts, asset, prev_state, state, rsi_zone, price, rsi "
         "FROM signal_history ORDER BY id DESC LIMIT ")
    try:
        if hasattr(broker, "_conn"):
            with broker._conn() as c, c.cursor() as cur:
                cur.execute(_HISTORY_SCHEMA_PG)
                cur.execute(q + "%s", (int(limit),))
                rows = cur.fetchall()
        elif hasattr(broker, "db"):
            with broker._lock:
                broker.db.execute(_HISTORY_SCHEMA_SQLITE)
                rows = broker.db.execute(q + "?", (int(limit),)).fetchall()
        else:
            return []
    except Exception:
        log.exception("signal history read failed")
        return []
    return [dict(zip(cols, r)) for r in rows]


def _message(name: str, sig: dict, prev_state: str | None) -> str:
    arrow = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[sig["state"]]
    lines = [
        f"{arrow} *{name.upper()} signal: {sig['state'].upper()}*"
        + (f" (was {prev_state})" if prev_state else ""),
        "",
        f"💰 {name}: ${sig['price']:,.2f}",
        f"📈 200-day average: ${sig['ma200']:,.2f} — price is "
        + ("ABOVE (long-term uptrend)" if sig["above_ma200"] else "BELOW (long-term downtrend)"),
        f"⚡ Momentum (MACD): {'positive' if sig['momentum_up'] else 'negative'}",
        f"🌡️ RSI: {sig['rsi']:.0f} ({sig['rsi_zone']})",
        "",
    ]
    if sig["state"] == "bullish":
        lines.append(f"📝 Simple meaning: trend and momentum both point UP — "
                     f"historically the better zone to be holding {name.lower()}.")
    elif sig["state"] == "bearish":
        lines.append(f"📝 Simple meaning: trend and momentum both point DOWN — "
                     f"historically the zone where holding {name.lower()} lost money.")
    else:
        lines.append("📝 Simple meaning: mixed picture — trend and momentum "
                     "disagree; no clear edge either way.")
    if sig["rsi_zone"] == "overbought":
        lines.append("⚠️ RSI overbought: the recent rally is stretched; "
                     "pullbacks are common from here.")
    elif sig["rsi_zone"] == "oversold":
        lines.append("⚠️ RSI oversold: the recent sell-off is stretched; "
                     "bounces are common from here.")
    lines.append("")
    lines.append("_Rule-based signal (200d MA + MACD + RSI), not financial advice._")
    return "\n".join(lines)


def _closes(asset: dict, cfg: dict) -> pd.Series:
    source = asset.get("source", "binance")
    if source == "yahoo":
        return yahoo.load_or_fetch_closes(asset["symbol"], cfg["data_paths"]["ohlcv"])
    end = pd.Timestamp.now(tz="UTC").floor("1D")
    start = end - pd.Timedelta(days=420)
    df = ohlcv.load_or_fetch(asset["symbol"], start.isoformat(), end.isoformat(),
                             cfg["data_paths"]["ohlcv"], timeframe="1d")
    return df["close"]


def check_asset(asset: dict, cfg: dict, broker) -> dict:
    name = asset["name"]
    close = _closes(asset, cfg)
    if len(close) < 150:
        return {"ok": False, "asset": name, "error": f"only {len(close)} bars of history"}
    sig = compute_signals(close)

    state_key = f"signal_state_{name.lower()}"
    prev_raw = _get_meta(broker, state_key)
    prev = json.loads(prev_raw) if prev_raw else {}
    changed = (prev.get("state"), prev.get("rsi_zone")) != (sig["state"], sig["rsi_zone"])
    if changed:
        try:
            send_alert(_message(name, sig, prev.get("state")))
        except Exception:  # alerting must never break the job
            log.exception("%s alert failed", name)
        _set_meta(broker, state_key, json.dumps(sig))
        ts = int(pd.Timestamp.now(tz="UTC").timestamp())
        record_history(broker, ts, name, prev.get("state"), sig)
    log.info("%s signal %s (changed=%s) price=%s", name, sig["state"], changed, sig["price"])
    return {"ok": True, "asset": name, "changed": changed, **sig}


def check_all(cfg: dict, broker) -> list[dict]:
    """Check every watched asset; one asset failing must not block the others."""
    assets = (cfg.get("signal_watch", {}) or {}).get("assets") or DEFAULT_ASSETS
    out = []
    for a in assets:
        try:
            out.append(check_asset(a, cfg, broker))
        except Exception as e:  # noqa: BLE001
            log.exception("signal check failed for %s", a.get("name"))
            out.append({"ok": False, "asset": a.get("name"), "error": str(e)})
    return out
