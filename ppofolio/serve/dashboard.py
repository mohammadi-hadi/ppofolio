"""Self-contained dashboard page served at GET /.

Zero external assets (no CDN, no fonts, no fetch beyond this service's own API):
inline CSS + JS, hand-rolled SVG line chart. Light/dark via prefers-color-scheme.
Colors follow the validated reference data-viz palette (series-1 blue = agent,
series-2 aqua = benchmark; status good/critical for the live/halted badge).
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ppofolio</title>
<style>
:root {
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --ink-1: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #1baf7a;
  --good: #0ca30c; --critical: #d03b3b; --warning: #fab219;
  --good-text: #006300;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #199e70;
    --good-text: #0ca30c;
  }
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--page); color: var(--ink-1);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 20px; }
.wrap { max-width: 1040px; margin: 0 auto; display: grid; gap: 14px; }
header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
header h1 { font-size: 18px; font-weight: 650; }
header .sub { color: var(--ink-2); font-size: 13px; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px;
  border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid var(--border); }
.badge .dot { width: 8px; height: 8px; border-radius: 50%; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .k { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
.tile .v { font-size: 22px; font-weight: 650; }
.tile .d { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.card h2 { font-size: 13px; font-weight: 600; color: var(--ink-2); margin-bottom: 10px; }
.legend { display: flex; gap: 14px; font-size: 12px; color: var(--ink-2); margin-bottom: 6px; }
.legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.chartwrap { position: relative; overflow-x: auto; }
svg text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--muted); }
.tooltip { position: absolute; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; padding: 6px 9px; font-size: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; white-space: nowrap; z-index: 5; }
.tooltip .t { color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-size: 12px; font-weight: 500;
  border-bottom: 1px solid var(--grid); padding: 4px 8px; }
td { padding: 5px 8px; border-bottom: 1px solid var(--grid); font-size: 13px; }
tr:last-child td { border-bottom: none; }
.num { text-align: right; }
.pos { color: var(--good-text); } .neg { color: var(--critical); }
.meta { color: var(--muted); font-size: 12px; }
a { color: var(--series-1); text-decoration: none; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title">ppofolio</h1>
    <span class="sub" id="subtitle"></span>
    <span class="badge" id="badge"><span class="dot"></span><span id="badge-text">…</span></span>
    <span class="meta" id="peer"></span>
  </header>

  <div class="tiles">
    <div class="tile"><div class="k">Equity</div><div class="v" id="t-equity">—</div><div class="d" id="t-cash"></div></div>
    <div class="tile"><div class="k">Return since reset</div><div class="v" id="t-return">—</div><div class="d" id="t-since"></div></div>
    <div class="tile"><div class="k">vs buy &amp; hold</div><div class="v" id="t-alpha">—</div><div class="d">same start, equal-weight, held</div></div>
    <div class="tile"><div class="k">Last run</div><div class="v" id="t-lastrun">—</div><div class="d" id="t-nextrun"></div></div>
  </div>

  <div class="card">
    <h2>Equity vs buy &amp; hold</h2>
    <div class="legend">
      <span><span class="sw" style="background:var(--series-1)"></span>Agent</span>
      <span><span class="sw" style="background:var(--series-2)"></span>Buy &amp; hold (equal weight)</span>
    </div>
    <div class="chartwrap" id="chartwrap">
      <svg id="chart" width="100%" height="260" role="img" aria-label="Equity curve versus buy and hold benchmark"></svg>
      <div class="tooltip" id="tooltip"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Positions</h2>
      <table id="positions"><thead><tr><th>Asset</th><th class="num">Units</th><th class="num">Weight</th></tr></thead><tbody></tbody></table>
      <div class="meta" style="margin-top:8px" id="halt-reason"></div>
    </div>
    <div class="card">
      <h2>Recent trades</h2>
      <table id="trades"><thead><tr><th>Time (UTC)</th><th>Asset</th><th class="num">Δ units</th><th class="num">Price</th><th class="num">Notional</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
  <div class="meta">Paper trading — simulated fills on real market data. Auto-refreshes every 60s. <span id="updated"></span></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const fmtUsd = (v) => "$" + Number(v).toLocaleString("en-US", {maximumFractionDigits: 0});
const fmtPct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
const fmtTs = (ts) => new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 16);
const ago = (sec) => sec < 90 ? "just now" : sec < 5400 ? Math.round(sec / 60) + "m ago" : sec < 129600 ? Math.round(sec / 3600) + "h ago" : Math.round(sec / 86400) + "d ago";

async function j(path) { const r = await fetch(path); if (!r.ok) throw new Error(path + " " + r.status); return r.json(); }

function drawChart(agent, bench) {
  const svg = $("chart"), wrap = $("chartwrap"), tip = $("tooltip");
  const W = Math.max(wrap.clientWidth, 320), H = 260, m = {t: 12, r: 14, b: 26, l: 58};
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";
  if (!agent.length) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle">no data yet</text>`; return; }
  const all = agent.concat(bench);
  const t0 = Math.min(...all.map(p => p.ts)), t1 = Math.max(...all.map(p => p.ts));
  let v0 = Math.min(...all.map(p => p.value)), v1 = Math.max(...all.map(p => p.value));
  if (v1 - v0 < 1e-9) { v0 -= 1; v1 += 1; }
  const pad = (v1 - v0) * 0.08; v0 -= pad; v1 += pad;
  const x = (ts) => t1 === t0 ? (m.l + W - m.r) / 2 : m.l + (ts - t0) / (t1 - t0) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v - v0) / (v1 - v0) * (H - m.t - m.b);
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
  // grid + y labels (4 ticks) — enough decimals that ticks stay distinct
  const dec = (v1 - v0) < 3000 ? 1 : 0;
  for (let i = 0; i <= 3; i++) {
    const v = v0 + (v1 - v0) * i / 3, yy = y(v);
    el("line", {x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: "var(--grid)", "stroke-width": 1});
    const t = el("text", {x: m.l - 8, y: yy + 4, "text-anchor": "end"});
    t.textContent = "$" + (v / 1000).toLocaleString("en-US", {minimumFractionDigits: dec, maximumFractionDigits: dec}) + "k";
  }
  // x labels (first, mid, last) — time-of-day when the span is under 2 days
  const short = (t1 - t0) < 2 * 86400;
  [t0, (t0 + t1) / 2, t1].forEach((ts, i) => {
    const t = el("text", {x: x(ts), y: H - 8, "text-anchor": i === 0 ? "start" : i === 2 ? "end" : "middle"});
    const iso = new Date(ts * 1000).toISOString();
    t.textContent = short ? iso.slice(5, 10) + " " + iso.slice(11, 16) : iso.slice(5, 10);
  });
  el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1});
  const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + x(p.ts).toFixed(1) + "," + y(p.value).toFixed(1)).join(" ");
  if (bench.length) el("path", {d: path(bench), fill: "none", stroke: "var(--series-2)", "stroke-width": 2, "stroke-linejoin": "round"});
  el("path", {d: path(agent), fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round"});
  // crosshair + tooltip
  const cross = el("line", {y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "3,3", visibility: "hidden"});
  const dotA = el("circle", {r: 4, fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden"});
  const dotB = el("circle", {r: 4, fill: "var(--series-2)", stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden"});
  const nearest = (pts, ts) => pts.reduce((a, b) => Math.abs(b.ts - ts) < Math.abs(a.ts - ts) ? b : a);
  svg.addEventListener("mousemove", (ev) => {
    const box = svg.getBoundingClientRect();
    const ts = t0 + (ev.clientX - box.left - m.l) / (W - m.l - m.r) * (t1 - t0);
    if (ts < t0 || ts > t1) return;
    const a = nearest(agent, ts), b = bench.length ? nearest(bench, ts) : null;
    cross.setAttribute("x1", x(a.ts)); cross.setAttribute("x2", x(a.ts)); cross.setAttribute("visibility", "visible");
    dotA.setAttribute("cx", x(a.ts)); dotA.setAttribute("cy", y(a.value)); dotA.setAttribute("visibility", "visible");
    if (b) { dotB.setAttribute("cx", x(b.ts)); dotB.setAttribute("cy", y(b.value)); dotB.setAttribute("visibility", "visible"); }
    tip.style.display = "block";
    tip.innerHTML = `<div class="t">${fmtTs(a.ts)}</div>Agent ${fmtUsd(a.value)}` + (b ? `<br>B&amp;H ${fmtUsd(b.value)}` : "");
    const tx = Math.min(ev.clientX - box.left + 14, W - 150);
    tip.style.left = tx + "px"; tip.style.top = (y(a.value) - 10) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none";
    [cross, dotA, dotB].forEach(e => e.setAttribute("visibility", "hidden")); });
}

async function refresh() {
  try {
    const [status, equity, trades, health] = await Promise.all([
      j("/status"), j("/equity"), j("/trades?limit=12"), j("/health")]);
    let bench = {curve: []};
    try { bench = await j("/benchmark"); } catch (e) {}

    const tf = status.timeframe === "1h" ? "hourly (long+short)" : "daily (long-only)";
    $("subtitle").textContent = tf + " · " + (status.config || "");
    document.title = "ppofolio · " + (status.timeframe === "1h" ? "hourly" : "daily");
    const halted = status.halted;
    const decom = status.decommissioned;
    $("badge-text").textContent = decom ? "DECOMMISSIONED" : halted ? "HALTED" : "LIVE";
    $("badge").querySelector(".dot").style.background = halted ? "var(--critical)" : "var(--good)";
    $("badge").style.color = halted ? "var(--critical)" : "var(--good-text)";

    const curve = equity.curve || [];
    const latest = curve.length ? curve[curve.length - 1] : null;
    const first = curve.length ? curve[0] : null;
    if (latest) {
      $("t-equity").textContent = fmtUsd(latest.value);
      $("t-cash").textContent = "cash " + fmtUsd(status.cash);
      const ret = latest.value / first.value - 1;
      $("t-return").textContent = fmtPct(ret);
      $("t-return").className = "v " + (ret >= 0 ? "pos" : "neg");
      $("t-since").textContent = "since " + fmtTs(first.ts);
      const age = Date.now() / 1000 - latest.ts;
      $("t-lastrun").textContent = ago(age);
      $("t-nextrun").textContent = status.timeframe === "1h" ? "runs hourly at :05" : "runs daily 00:05 UTC";
      const bcurve = bench.curve || [];
      if (bcurve.length) {
        const bret = bcurve[bcurve.length - 1].value / bcurve[0].value - 1;
        const alpha = ret - bret;
        $("t-alpha").textContent = fmtPct(alpha);
        $("t-alpha").className = "v " + (alpha >= 0 ? "pos" : "neg");
      }
      drawChart(curve, bcurve);
    }

    const ptb = $("positions").querySelector("tbody");
    ptb.innerHTML = "";
    const w = latest ? latest.weights : [];
    status.assets.forEach((a, i) => {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${a}</td><td class="num">${Number(status.holdings[i]).toFixed(6)}</td>` +
        `<td class="num">${w[i] !== undefined ? fmtPct(w[i]) : "—"}</td>`;
      ptb.appendChild(row);
    });
    $("halt-reason").textContent = halted ? ("halt reason: " + (status.halt_reason || "unknown") +
      " — clear via POST /reset-halt") : "";

    const ttb = $("trades").querySelector("tbody");
    ttb.innerHTML = "";
    (trades.trades || []).forEach(t => {
      const row = document.createElement("tr");
      const cls = t.delta_units >= 0 ? "pos" : "neg";
      row.innerHTML = `<td>${fmtTs(t.ts)}</td><td>${t.asset}</td>` +
        `<td class="num ${cls}">${t.delta_units >= 0 ? "+" : ""}${Number(t.delta_units).toFixed(5)}</td>` +
        `<td class="num">${Number(t.exec_price).toLocaleString("en-US", {maximumFractionDigits: 2})}</td>` +
        `<td class="num">${fmtUsd(Math.abs(t.notional))}</td>`;
      ttb.appendChild(row);
    });
    if (!(trades.trades || []).length) ttb.innerHTML = '<tr><td colspan="5" class="meta">no trades yet</td></tr>';

    if (health.peer_url) $("peer").innerHTML = `<a href="${health.peer_url}">→ sibling service</a>`;
    $("updated").textContent = "Updated " + new Date().toISOString().replace("T", " ").slice(0, 19) + "Z";
  } catch (e) {
    $("updated").textContent = "refresh failed: " + e.message;
  }
}
refresh();
setInterval(refresh, 60000);
addEventListener("resize", () => refresh());
</script>
</body>
</html>
"""
