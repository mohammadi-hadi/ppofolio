"""FastAPI service: dashboard at /, JSON API for state.

A background scheduler (APScheduler) fires `run_once` at 00:05 UTC daily, or at
:05 every hour when the config's timeframe is 1h.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from ppofolio.data import ohlcv
from ppofolio.live.alerts import send as send_alert
from ppofolio.live.kill_switch import (_get_meta, clear_halt, is_decommissioned, is_halted,
                                  maybe_send_halted_reminder, set_halt)
from ppofolio.live.loop import run_once
from ppofolio.live.pg_broker import make_broker
from ppofolio.live.watchdog import run_watchdog
from ppofolio.serve.dashboard import DASHBOARD_HTML
from ppofolio.signals.watch import check_all, read_history
from ppofolio.utils import ROOT, ensure_dir, load_config

log = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

CONFIG_PATH = os.environ.get("CONFIG", "configs/base.yaml")
MODEL_PATH = Path(ROOT) / os.environ.get("MODEL", "models/policy.zip")

cfg = load_config(CONFIG_PATH)
DB_PATH = Path(ROOT) / "data" / "paper.db"
ensure_dir(DB_PATH.parent)
ASSETS = [a["symbol"].split("/")[0] for a in cfg["assets"]]
TIMEFRAME = cfg.get("timeframe", "1d")

scheduler = BackgroundScheduler(timezone="UTC")

# One rebalance/reset at a time: the cron job, the watchdog, /run-now and
# /admin/reset all mutate the same broker and must never overlap.
RUN_LOCK = threading.Lock()

_BROKER = None
_BROKER_LOCK = threading.Lock()


def get_broker():
    """Lazily construct the broker with retries so a Postgres blip at boot
    degrades /health instead of crash-looping the container."""
    global _BROKER
    if _BROKER is not None:
        return _BROKER
    with _BROKER_LOCK:
        if _BROKER is not None:
            return _BROKER
        last_err = None
        for attempt in range(3):
            try:
                _BROKER = make_broker(
                    assets=ASSETS,
                    initial_cash=cfg["initial_cash"],
                    fee_bps=cfg["fee_bps"],
                    slippage_std=[a["slippage_std"] for a in cfg["assets"]],
                    sqlite_fallback_path=DB_PATH,
                    borrow_rate_daily=float(cfg.get("borrow_rate_daily", 0.0)),
                )
                return _BROKER
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("broker init failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise last_err


def _require_admin(token: str | None) -> None:
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN not configured on this service")
    if token != expected:
        raise HTTPException(status_code=403, detail="invalid admin token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_broker()
    except Exception as e:  # noqa: BLE001
        log.error("broker unavailable at boot, will retry on demand: %s", e)
    if TIMEFRAME == "1h":
        scheduler.add_job(_job, CronTrigger(minute=5, timezone="UTC"), id="hourly-rebalance")
    else:
        scheduler.add_job(_job, CronTrigger(hour=0, minute=5, timezone="UTC"), id="daily-rebalance")
        scheduler.add_job(_watchdog_job, CronTrigger(minute=7, timezone="UTC"), id="hourly-watchdog")
    if cfg.get("signal_watch", {}).get("enabled"):
        scheduler.add_job(_signals_job, CronTrigger(hour=0, minute=20, timezone="UTC"), id="signal-watch")
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


def _watchdog_job() -> None:
    if not RUN_LOCK.acquire(blocking=False):
        log.info("watchdog skipped: another run in progress")
        return
    try:
        broker = get_broker()
        r = run_watchdog(CONFIG_PATH, broker)
        log.info("watchdog %s", r)
        maybe_send_halted_reminder(broker)
    except Exception as e:  # noqa: BLE001
        log.exception("watchdog failed: %s", e)
    finally:
        RUN_LOCK.release()


def _signals_job() -> None:
    try:
        for r in check_all(cfg, get_broker()):
            log.info("signal watch %s", r)
    except Exception as e:  # noqa: BLE001
        log.exception("signal watch failed: %s", e)


app = FastAPI(title="ppofolio", lifespan=lifespan)


@app.get("/signals")
def signals():
    """Rule-based signals for all watched assets (recomputed on request; alerts only on change)."""
    if not cfg.get("signal_watch", {}).get("enabled"):
        return {"ok": False, "error": "signal_watch not enabled on this service"}
    return {"signals": check_all(cfg, get_broker())}


@app.get("/signals/history")
def signals_history(limit: int = 100):
    """Signal state CHANGES over time (what was alerted), newest first."""
    if not cfg.get("signal_watch", {}).get("enabled"):
        return {"ok": False, "error": "signal_watch not enabled on this service"}
    return {"history": read_history(get_broker(), limit=limit)}


@app.get("/gold")
def gold():
    """Back-compat: the gold entry from /signals."""
    if not cfg.get("signal_watch", {}).get("enabled"):
        return {"ok": False, "error": "signal_watch not enabled on this service"}
    for r in check_all(cfg, get_broker()):
        if r.get("asset") == "Gold":
            return r
    return {"ok": False, "error": "gold not among watched assets"}


def _has_policy() -> bool:
    if MODEL_PATH.exists():
        return True
    prefix = cfg.get("model_prefix", "policy_s")
    return any((Path(ROOT) / "models").glob(f"{prefix}*.zip"))


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    checks = {"model_present": _has_policy(), "db": False, "scheduler": scheduler.running}
    last_run_age = None
    try:
        broker = get_broker()
        checks["db"] = True
        curve = broker.equity_curve(limit=1)
        if curve:
            last_run_age = int(time.time() - curve[-1]["ts"])
    except Exception as e:  # noqa: BLE001
        checks["db_error"] = str(e)
    return {
        "ok": checks["db"] and checks["model_present"],
        "assets": ASSETS,
        "timeframe": TIMEFRAME,
        "last_run_age_seconds": last_run_age,
        "peer_url": os.environ.get("PEER_URL"),
        **checks,
    }


@app.get("/status")
def status():
    broker = get_broker()
    curve = broker.equity_curve(limit=1)
    latest = curve[-1] if curve else None
    return {
        "cash": broker.cash,
        "holdings": broker.holdings().tolist(),
        "latest": latest,
        "assets": ASSETS,
        "halted": is_halted(broker),
        "halt_reason": _get_meta(broker, "halt_reason"),
        "decommissioned": is_decommissioned(broker),
        "config": CONFIG_PATH,
        "timeframe": TIMEFRAME,
    }


@app.get("/trades")
def trades(limit: int = 50):
    return {"trades": get_broker().recent_trades(limit)}


@app.get("/equity")
def equity(since: int | None = None, limit: int | None = None):
    return {"curve": get_broker().equity_curve(since=since, limit=limit)}


@app.get("/benchmark")
def benchmark():
    """Equal-weight buy-and-hold-no-rebalance from the account's first equity point."""
    try:
        broker = get_broker()
        curve = broker.equity_curve()
        if not curve:
            return {"curve": []}
        t0, e0 = int(curve[0]["ts"]), float(curve[0]["value"])
        now = pd.Timestamp.now(tz="UTC")
        start = pd.Timestamp(t0, unit="s", tz="UTC").floor("1D") - pd.Timedelta(days=3)
        end = now.floor("1D")
        closes = {}
        for a in cfg["assets"]:
            df = ohlcv.load_or_fetch(a["symbol"], start.isoformat(), end.isoformat(),
                                     cfg["data_paths"]["ohlcv"], timeframe="1d")
            closes[a["symbol"]] = df["close"]
        prices = pd.DataFrame(closes).sort_index().ffill().dropna()
        if prices.empty:
            return {"curve": []}
        # Entry price = last daily close at/before the account start time.
        entry = prices[prices.index <= pd.Timestamp(t0, unit="s", tz="UTC")]
        p0 = (entry.iloc[-1] if not entry.empty else prices.iloc[0])
        units = (e0 / len(p0)) / p0
        # A daily bar stamped D closes at D+1 00:00 — plot values at close time,
        # strictly after the account start.
        vals = (prices * units).sum(axis=1)
        pts = [{"ts": int(ts.timestamp()) + 86400, "value": float(v)}
               for ts, v in vals.items() if int(ts.timestamp()) + 86400 > t0]
        pts.insert(0, {"ts": t0, "value": e0})
        # Mark-to-now so the benchmark line reaches the agent's latest point.
        latest_ts = int(curve[-1]["ts"])
        if latest_ts > pts[-1]["ts"]:
            pts.append({"ts": latest_ts, "value": float((prices.iloc[-1] * units).sum())})
        dedup = {p["ts"]: p for p in pts}
        return {"curve": [dedup[k] for k in sorted(dedup)]}
    except Exception as e:  # noqa: BLE001
        log.warning("benchmark failed: %s", e)
        return {"curve": [], "error": "benchmark failed; see the service log"}


@app.get("/halt-status")
def halt_status():
    return {"halted": is_halted(get_broker())}


@app.post("/admin/halt")
def admin_halt(reason: str = "manual halt", decommission: bool = False,
               x_admin_token: str | None = Header(default=None)):
    """Engage the halt (optionally permanently: decommission). Positions are
    liquidated on the next scheduled/manual run via the halted path."""
    _require_admin(x_admin_token)
    broker = get_broker()
    set_halt(broker, reason, decommission=decommission)
    try:
        send_alert(("🛑 *DECOMMISSIONED*: " if decommission else "⏸️ *Halted*: ") + reason)
    except Exception:
        pass
    return {"ok": True, "halted": True, "decommissioned": is_decommissioned(broker), "reason": reason}


@app.post("/reset-halt")
def reset_halt(x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    broker = get_broker()
    clear_halt(broker)
    return {"ok": True, "halted": is_halted(broker)}


@app.post("/admin/reset")
def admin_reset(x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a run is in progress, retry shortly")
    try:
        broker = get_broker()
        broker.reset(cfg["initial_cash"])
        try:
            send_alert(f"🔄 *Paper account reset* — fresh start at {cfg['initial_cash']:,.0f}. "
                       "History archived, halt cleared.")
        except Exception:
            pass
        return {"ok": True, "cash": broker.cash, "halted": is_halted(broker)}
    finally:
        RUN_LOCK.release()


@app.post("/run-now")
def run_now(x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    if not _has_policy():
        return {"ok": False, "error": "no model found for prefix "
                f"{cfg.get('model_prefix', 'policy_s')} (or {MODEL_PATH})"}
    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a run is in progress, retry shortly")
    try:
        result = run_once(cfg, MODEL_PATH, get_broker())
        return {"ok": "error" not in result, "result": result}
    except Exception as e:  # noqa: BLE001
        log.exception("run-now failed: %s", e)
        return {"ok": False, "error": "run failed; see the service log"}
    finally:
        RUN_LOCK.release()


def _job() -> None:
    if not RUN_LOCK.acquire(blocking=False):
        log.warning("scheduled run skipped: another run in progress")
        return
    try:
        if not _has_policy():
            log.warning("no model found — skipping scheduled run")
            return
        broker = get_broker()
        if is_decommissioned(broker):
            # Permanently flat: skip data fetching / model inference entirely.
            if float(np.abs(broker.holdings()).sum()) < 1e-9:
                log.info("decommissioned and flat — nothing to do")
                return
        r = run_once(cfg, MODEL_PATH, broker)
        if r.get("error"):
            log.error("scheduled run aborted: %s", r["error"])
            try:
                send_alert(f"⚠️ *Scheduled run aborted*: {r['error']}")
            except Exception:
                pass
        else:
            log.info("scheduled run equity=%.2f weights=%s", r["equity"], r["weights"])
        maybe_send_halted_reminder(broker)
    except Exception as e:  # noqa: BLE001
        log.exception("scheduled job failed: %s", e)
        try:
            send_alert(f"⚠️ *Scheduled run FAILED*: {e}")
        except Exception:
            pass
    finally:
        RUN_LOCK.release()
