"""Build the unified multi-asset feature panel and cache it as parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ppofolio.data import coinmetrics, macro, news, ohlcv
from ppofolio.features.assemble import build_panel, finalize
from ppofolio.features.technical import add_technical
from ppofolio.utils import ROOT, ensure_dir, load_config


def asset_slug(symbol: str) -> str:
    return symbol.split("/")[0].lower()


def build(cfg: dict, start: str, end: str) -> pd.DataFrame:
    timeframe = cfg.get("timeframe", "1d")
    bars_per_day = int(cfg.get("bars_per_day", 1))
    per_asset: dict[str, pd.DataFrame] = {}
    for a in cfg["assets"]:
        sym = a["symbol"]
        slug = asset_slug(sym)
        print(f"[build] OHLCV {sym} @ {timeframe}")
        px = ohlcv.load_or_fetch(sym, start, end, cfg["data_paths"]["ohlcv"], timeframe=timeframe)
        px = add_technical(px, bars_per_day=bars_per_day)

        print(f"[build] on-chain {a['coinmetrics']} (daily -> broadcast)")
        oc = coinmetrics.load_or_fetch(a["coinmetrics"], start, end, cfg["data_paths"]["onchain"])
        if timeframe != "1d" and not oc.empty:
            oc = macro.broadcast_to_index(oc, px.index)

        if cfg.get("news", {}).get("enabled"):
            print(f"[build] news {slug}")
            q = cfg["news"]["queries"][[asset_slug(x["symbol"]) for x in cfg["assets"]].index(slug)]
            sn = news.load_or_fetch(q, slug, start, end, cfg["data_paths"]["news"], cfg["news"]["max_per_day"])
        else:
            idx = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
            sn = pd.DataFrame(index=idx, data={"sent_mean": 0.0, "sent_count": 0})

        merged = px.join(oc, how="left").join(sn, how="left")
        per_asset[slug.upper()] = merged

    # Market-wide macro features (broadcast across assets and to panel grid)
    print("[build] macro (fng, btc dominance, funding rate)")
    macro_df = macro.load_or_fetch_macro(start, end, cfg["data_paths"].get("macro", "data/macro"))
    if timeframe != "1d" and not macro_df.empty:
        # target index: union of per-asset OHLCV timestamps (will match panel index after join)
        target = per_asset[list(per_asset.keys())[0]].index
        macro_df = macro.broadcast_to_index(macro_df, target)

    panel = build_panel(
        per_asset,
        tech_cols=cfg["features"]["technical"],
        onchain_cols=cfg["features"]["onchain"],
        sent_cols=cfg["features"]["sentiment"],
    )
    # Broadcast macro across all assets (same value per date, replicated per asset).
    # They appear with "MACRO__" prefix (they're not asset-specific) and enter the obs
    # via the feature assembly below.
    macro_df.columns = [f"MACRO__{c}" for c in macro_df.columns]
    panel = panel.join(macro_df, how="left")
    panel = finalize(panel)
    return panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-01-01")
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = build(cfg, args.start, args.end)

    out_path = Path(ROOT) / cfg["data_paths"]["processed"]
    ensure_dir(out_path.parent)
    panel.to_parquet(out_path)
    print(f"[build] wrote {out_path}  shape={panel.shape}  range=[{panel.index.min().date()} .. {panel.index.max().date()}]")
    nulls = panel.isna().sum().sum()
    print(f"[build] total NaNs: {nulls}")


if __name__ == "__main__":
    main()
