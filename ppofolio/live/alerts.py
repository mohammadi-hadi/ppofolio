"""Trade-alert notifications.

Sends one message per executed trade to Telegram and/or a Discord webhook
(whichever env vars are set). Silent no-op if neither is configured.

Env vars:
  TELEGRAM_BOT_TOKEN  + TELEGRAM_CHAT_ID   -> Telegram
  DISCORD_WEBHOOK_URL                      -> Discord
  ALERTS_EXCHANGE     (display only, default "BINANCE")
  ALERTS_TIMEFRAME    (display only, default "1D")
  ALERTS_TAG          (short tag echoed in the alert, default "M1")
"""
from __future__ import annotations

import logging
import os
import queue
import time
from threading import Lock, Thread

import requests

log = logging.getLogger(__name__)

SIDE_EMOJI = {"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "⚪ HOLD"}

# In-memory throttle: {(asset, sign): last_sent_epoch}. Prevents alert floods at hourly cadence.
_THROTTLE: dict[tuple[str, int], float] = {}
_THROTTLE_LOCK = Lock()

# Alerts go through a background queue so a slow Telegram/Discord round-trip can
# never block the trading path.
_QUEUE: "queue.Queue[tuple[str, str | None, int | None]]" = queue.Queue(maxsize=200)
_WORKER_STARTED = False
_WORKER_LOCK = Lock()


def _cooldown_sec() -> int:
    try:
        return int(os.environ.get("ALERTS_COOLDOWN_SEC", "1800"))
    except ValueError:
        return 1800


def _throttled(asset: str, sign: int) -> bool:
    cd = _cooldown_sec()
    if cd <= 0:
        return False
    with _THROTTLE_LOCK:
        return time.time() - _THROTTLE.get((asset, sign), 0.0) < cd


def _mark_sent(asset: str, sign: int) -> None:
    with _THROTTLE_LOCK:
        _THROTTLE[(asset, sign)] = time.time()


def _worker() -> None:
    while True:
        msg, asset, sign = _QUEUE.get()
        try:
            ok_tg = _send_telegram(msg)
            ok_dc = _send_discord(msg)
            # Mark the cooldown only when something actually went out, so a failed
            # send doesn't suppress the next (possibly successful) one.
            if asset is not None and sign is not None and (ok_tg or ok_dc):
                _mark_sent(asset, sign)
        except Exception as e:  # noqa: BLE001
            log.warning("alert worker error: %s", e)
        finally:
            _QUEUE.task_done()


def _ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            Thread(target=_worker, daemon=True, name="alerts-sender").start()
            _WORKER_STARTED = True


def _side(delta_units: float, prev_units: float) -> tuple[str, str]:
    """Return (side_label, plain_meaning)."""
    if abs(delta_units) < 1e-12:
        return "hold", "No change — position held"
    new_units = prev_units + delta_units
    # Reversal if sign flipped
    if prev_units != 0 and (prev_units > 0) != (new_units > 0) and new_units != 0:
        arrow = "long → short" if prev_units > 0 else "short → long"
        return "sell" if delta_units < 0 else "buy", f"Position reversal: {arrow}"
    if prev_units == 0:
        return ("buy" if delta_units > 0 else "sell",
                "Opened a new " + ("BUY" if delta_units > 0 else "SELL") + " position")
    if (delta_units > 0) == (prev_units > 0):
        return ("buy" if delta_units > 0 else "sell", "Added to existing position")
    return ("sell" if delta_units < 0 else "buy", "Reduced existing position")


def format_alert(*, asset: str, price: float, delta_units: float, prev_units: float,
                 new_units: float, weight: float, ts_iso: str,
                 exchange: str | None = None, timeframe: str | None = None,
                 tag: str | None = None) -> str:
    side_label, meaning = _side(delta_units, prev_units)
    head = SIDE_EMOJI.get(side_label, side_label.upper())
    ex = exchange or os.environ.get("ALERTS_EXCHANGE", "BINANCE")
    tf = timeframe or os.environ.get("ALERTS_TIMEFRAME", "1D")
    tag = tag or os.environ.get("ALERTS_TAG", "M1")
    lines = [
        "📊 *Trade Alert*",
        "",
        f"📝 *Simple meaning:* {meaning}",
        "",
        f"🔄 *Action:* {head}",
        f"📈 *Symbol:* {asset}",
        f"🏦 *Exchange:* {ex}",
        f"💰 *Price:* {price:,.2f}",
        f"📦 *Contracts:* {delta_units:+.6f}  (this order)",
        f"📌 *Position Size:* {new_units:+.6f}  (after)",
        f"⚖️ *Weight:* {weight*100:+.2f}%",
        f"⏰ *Time:* {ts_iso}",
        f"🕐 *Timeframe:* {tf}",
        f"🚨 *Message:* {tag}",
    ]
    return "\n".join(lines)


def _send_telegram(msg: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True},
            timeout=10,
        )
        if r.status_code != 200:
            log.warning("telegram send failed: %s %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("telegram exception: %s", e)
        return False


def _send_discord(msg: str) -> bool:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return False
    try:
        r = requests.post(url, json={"content": msg}, timeout=10)
        if r.status_code >= 300:
            log.warning("discord send failed: %s %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("discord exception: %s", e)
        return False


def send(msg: str, asset: str | None = None, sign: int | None = None) -> dict:
    """Queue for delivery to whichever channels are configured. Always logs.

    If asset+sign provided, apply per-(asset, sign) cooldown (default 30 min)
    so we don't flood Telegram when the hourly loop emits repeat trades.
    Delivery is asynchronous (background thread) — this call never blocks.
    """
    log.info("ALERT: %s", msg.replace("\n", " | "))
    if asset is not None and sign is not None and _throttled(asset, sign):
        log.info("alert throttled for %s/%s", asset, sign)
        return {"queued": False, "throttled": True}
    _ensure_worker()
    try:
        _QUEUE.put_nowait((msg, asset, sign))
        return {"queued": True, "throttled": False}
    except queue.Full:
        log.warning("alert queue full — dropping alert")
        return {"queued": False, "throttled": False}
