"""Daily-loss kill switch.

If equity drops more than max_daily_loss in one day (or cumulative_loss from peak),
engage the kill switch: all future rebalances are forced to 100% cash, a
'HALTED' alert is sent once, and a flag in broker meta persists until manually
cleared via POST /reset-halt.

Config (in configs/*.yaml under `kill_switch:`):
    max_daily_loss:      0.05   # -5% day over day
    max_drawdown_peak:   0.15   # -15% from all-time peak
    enabled:             true
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ppofolio.live.alerts import send as send_alert

log = logging.getLogger(__name__)

REMINDER_INTERVAL_SEC = 24 * 3600


def basket_recovery_ok(closes: np.ndarray, window: int = 10) -> bool | None:
    """True when the equal-weight asset basket has stabilized: basket index above
    its trailing `window`-bar mean. Used to gate kill-switch auto-resume — the
    pure 48h timer re-bought into falling markets (test window: 4 halts made the
    system 4.3pts worse than the raw policy).

    `closes` is (T, n_assets); each asset is normalized by its first close so no
    single expensive asset dominates the basket. Returns None (unknown) with
    insufficient history.
    """
    closes = np.asarray(closes, dtype=float)
    if closes.ndim == 1:
        closes = closes[:, None]
    if len(closes) < window + 1 or not np.isfinite(closes).all():
        return None
    base = closes[0]
    if (base <= 0).any():
        return None
    idx = (closes / base).mean(axis=1)
    return bool(idx[-1] > idx[-window:].mean())


def _get_meta(broker, key: str) -> str | None:
    """Read a key from broker meta. Works with both SQLite and Postgres brokers."""
    try:
        # PostgresBroker
        if hasattr(broker, "_conn"):
            with broker._conn() as c, c.cursor() as cur:
                cur.execute("SELECT v FROM broker_meta WHERE k=%s", (key,))
                row = cur.fetchone()
                return row[0] if row else None
        # SQLite PaperBroker
        if hasattr(broker, "db"):
            with broker._lock:
                row = broker.db.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
                return row[0] if row else None
    except Exception as e:
        log.warning("kill_switch meta read failed: %s", e)
    return None


def _set_meta(broker, key: str, value: str) -> None:
    try:
        if hasattr(broker, "_conn"):
            with broker._conn() as c, c.cursor() as cur:
                cur.execute(
                    "INSERT INTO broker_meta(k, v) VALUES (%s, %s) "
                    "ON CONFLICT (k) DO UPDATE SET v=EXCLUDED.v", (key, value))
            return
        if hasattr(broker, "db"):
            with broker._lock:
                broker.db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (key, value))
                broker.db.commit()
    except Exception as e:
        log.warning("kill_switch meta write failed: %s", e)


def is_halted(broker) -> bool:
    return _get_meta(broker, "halted") == "1"


def is_decommissioned(broker) -> bool:
    """A decommissioned service is permanently flat: no auto-resume, no reminders."""
    return _get_meta(broker, "decommissioned") == "1"


def set_halt(broker, reason: str, decommission: bool = False) -> None:
    _set_meta(broker, "halted", "1")
    _set_meta(broker, "halt_reason", reason)
    _set_meta(broker, "halted_at", str(int(time.time())))
    if decommission:
        _set_meta(broker, "decommissioned", "1")


def clear_halt(broker) -> None:
    _set_meta(broker, "halted", "0")
    _set_meta(broker, "decommissioned", "0")
    try:
        send_alert("✅ *Kill switch cleared* — trading resumes on the next scheduled run.")
    except Exception:
        pass


def maybe_send_halted_reminder(broker) -> bool:
    """While halted, re-alert once per 24h so a halt can never go silently stale."""
    if not is_halted(broker) or is_decommissioned(broker):
        return False
    now = time.time()
    last = _get_meta(broker, "last_halt_reminder")
    if last is not None:
        try:
            if now - float(last) < REMINDER_INTERVAL_SEC:
                return False
        except ValueError:
            pass
    _set_meta(broker, "last_halt_reminder", str(now))
    try:
        send_alert(
            "⏸️ *Still HALTED* — the kill switch is engaged and all rebalances go to cash.\n"
            "Clear it with POST /reset-halt (or POST /admin/reset for a full account reset)."
        )
    except Exception:
        pass
    return True


def check_and_update(broker, new_equity: float, cfg: dict,
                     recovery_ok: bool | None = None) -> tuple[bool, str]:
    """Return (halt_now, reason). Engages halt if thresholds crossed.

    `recovery_ok` (from basket_recovery_ok) gates auto-resume when the config
    sets kill_switch.resume_requires_recovery: after the cooldown the halt only
    clears once the market has stopped falling. None (unknown) fails open to the
    plain timer so a data hiccup can't leave the account halted forever.
    """
    ks = cfg.get("kill_switch", {})
    if not ks.get("enabled", True):
        return False, "disabled"
    if is_halted(broker):
        if is_decommissioned(broker):
            return True, "decommissioned"
        # Cooldown auto-resume: a halt is flash-crash protection, not a shutdown.
        # The manual-reset-only design left the account dead for six weeks once.
        auto_h = float(ks.get("auto_resume_hours", 0.0))
        halted_at = _get_meta(broker, "halted_at")
        if auto_h > 0 and halted_at is not None:
            try:
                if time.time() - float(halted_at) >= auto_h * 3600:
                    if bool(ks.get("resume_requires_recovery", False)) and recovery_ok is False:
                        return True, "halted: cooldown elapsed, awaiting basket recovery"
                    _set_meta(broker, "halted", "0")
                    # Re-base the drawdown peak so the peak rule doesn't re-trip
                    # instantly on the pre-halt high.
                    _set_meta(broker, "peak_since", str(int(time.time())))
                    try:
                        send_alert(f"▶️ *Auto-resumed* after {auto_h:.0f}h halt cooldown. "
                                   "Kill switch re-armed against the new peak.")
                    except Exception:
                        pass
                    log.warning("kill switch auto-resumed after %.0fh", auto_h)
                    return False, "auto-resumed"
            except ValueError:
                pass
        return True, "already-halted"

    max_daily = float(ks.get("max_daily_loss", 0.05))
    max_peak = float(ks.get("max_drawdown_peak", 0.15))

    # "Daily loss" compares to equity window_hours ago by TIMESTAMP (default 24h),
    # so extra equity rows from /run-now or the watchdog can't shrink the window.
    # Falls back to row-count window_bars for brokers without the targeted queries.
    window_hours = float(ks.get("window_hours", 24.0))
    peak_since = _get_meta(broker, "peak_since")
    prev_eq = peak = None
    if hasattr(broker, "equity_at_or_before"):
        cutoff = int(time.time() - window_hours * 3600)
        prev_eq = broker.equity_at_or_before(cutoff)
        peak = broker.equity_peak(since=int(float(peak_since)) if peak_since else None)
    if prev_eq is None or peak is None:
        curve = broker.equity_curve()
        window_bars = int(ks.get("window_bars", 1))
        if prev_eq is None:
            if len(curve) >= window_bars:
                prev_eq = curve[-window_bars]["value"]
            elif curve:
                prev_eq = curve[-1]["value"]
            else:
                prev_eq = new_equity
        if peak is None:
            peak = max([c["value"] for c in curve] + [new_equity])
    peak = max(peak, new_equity)

    daily_loss = (new_equity - prev_eq) / max(prev_eq, 1e-9)
    peak_loss = (new_equity - peak) / max(peak, 1e-9)

    trip_reason: str | None = None
    if daily_loss <= -max_daily:
        trip_reason = f"daily loss {daily_loss:+.2%} <= -{max_daily:.0%}"
    elif peak_loss <= -max_peak:
        trip_reason = f"peak drawdown {peak_loss:+.2%} <= -{max_peak:.0%}"

    if trip_reason is not None:
        _set_meta(broker, "halted", "1")
        _set_meta(broker, "halt_reason", trip_reason)
        _set_meta(broker, "halted_at", str(int(time.time())))
        try:
            send_alert(
                "🚨 *KILL SWITCH ENGAGED*\n\n"
                f"Equity: {new_equity:,.2f}\n"
                f"Trigger: {trip_reason}\n\n"
                "All future rebalances → 100% cash until manually cleared "
                "(POST /reset-halt)."
            )
        except Exception:
            pass
        log.warning("kill switch engaged: %s", trip_reason)
        return True, trip_reason
    return False, f"ok daily={daily_loss:+.2%} peak={peak_loss:+.2%}"
