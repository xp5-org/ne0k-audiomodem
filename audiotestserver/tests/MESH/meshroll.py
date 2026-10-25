# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshroll.py — piano-roll / call-flow renderer for mesh sim event streams.

Reads an events.jsonl (meshevents.py schema, from either sim tier) and writes a
single self-contained HTML file: no CDN, no server, everything inlined.

Layout (as specified):
  * TIME runs down the page (vertical axis, top = t0).
  * each node is a horizontal LANE; a tx renders as a filled block spanning its
    airtime; a delivery renders as an ARROW from the sender's lane to the
    receiver's lane at the reception time — a call-flow / sequence-diagram edge.
  * rx failures, discoveries, acks, ranges and carrier-defers get their own
    marks; hover any mark for a tooltip.
  * a RECEIPT panel at the top summarises the run: per-node counts, the
    who-discovered-whom matrix, collision rate, link-SNR stats, ranging results.

Usage:
  python tests/meshroll.py events.jsonl [-o run.html]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys


# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

import meshevents as ev  # noqa: E402


# ── summary / receipt ────────────────────────────────────────────────────────
def build_summary(meta: dict, records: list[dict]) -> dict:
    nodes: dict[int, dict] = {}
    for r in records:
        if r["kind"] == ev.KIND_BOOT:
            d = r.get("detail") or {}
            nodes[r["node"]] = {"profile": d.get("profile", "clean"),
                                "pos": d.get("pos"), "drift_ppm": d.get("drift_ppm", 0)}
    # ensure every node that appears at all is present
    for r in records:
        nodes.setdefault(r["node"], {"profile": "clean", "pos": None, "drift_ppm": 0})

    ids = sorted(nodes)
    per = {nid: {"tx": 0, "rx_ok": 0, "rx_fail": 0, "discovered": 0,
                 "collision": 0, "low_sinr": 0, "deaf": 0} for nid in ids}
    disc_matrix = {a: set() for a in ids}   # a discovered {b, …}
    snrs: list[float] = []
    ranges: list[dict] = []

    for r in records:
        n, k = r["node"], r["kind"]
        if n not in per:
            continue
        if k == ev.KIND_TX_START:
            per[n]["tx"] += 1
        elif k == ev.KIND_RX_OK:
            per[n]["rx_ok"] += 1
            if r.get("snr_db") is not None:
                snrs.append(r["snr_db"])
        elif k == ev.KIND_RX_FAIL:
            per[n]["rx_fail"] += 1
            reason = r.get("reason")
            if reason in ("collision", "low_sinr", "deaf"):
                per[n][reason] += 1
        elif k == ev.KIND_DISCOVERED:
            per[n]["discovered"] += 1
            if r.get("peer") is not None:
                disc_matrix[n].add(r["peer"])
        elif k == ev.KIND_RANGED:
            d = r.get("detail") or {}
            ranges.append({"a": n, "b": r.get("peer"),
                           "distance_m": d.get("distance_m")})

    total_rx = sum(p["rx_ok"] + p["rx_fail"] for p in per.values())
    total_coll = sum(p["collision"] for p in per.values())
    full = [nid for nid in ids if len(disc_matrix[nid]) == len(ids) - 1]

    return {
        "ids": ids,
        "nodes": nodes,
        "per": per,
        "disc_matrix": {a: sorted(b) for a, b in disc_matrix.items()},
        "full_discovery": full,
        "collision_rate": (total_coll / total_rx) if total_rx else 0.0,
        "snr_min": min(snrs) if snrs else None,
        "snr_mean": (sum(snrs) / len(snrs)) if snrs else None,
        "snr_max": max(snrs) if snrs else None,
        "ranges": ranges,
        "t_end": max((r["t"] for r in records), default=0.0),
        "n_events": len(records),
    }


# ── HTML ─────────────────────────────────────────────────────────────────────
PROFILE_COLOR = {"clean": "#3fb950", "noisy": "#d29922", "very_noisy": "#f85149"}


def _receipt_html(meta: dict, s: dict) -> str:
    ids = s["ids"]
    rows = []
    for nid in ids:
        p = s["per"][nid]
        prof = s["nodes"][nid]["profile"]
        c = PROFILE_COLOR.get(prof, "#888")
        ndisc = len(s["disc_matrix"][nid])
        rows.append(
            f"<tr><td><span class='dot' style='background:{c}'></span>{nid}</td>"
            f"<td>{prof}</td><td>{ndisc}/{len(ids)-1}</td><td>{p['tx']}</td>"
            f"<td>{p['rx_ok']}</td><td>{p['rx_fail']}</td>"
            f"<td>{p['collision']}</td><td>{p['low_sinr']}</td></tr>")
    per_table = (
        "<table class='rc'><thead><tr><th>node</th><th>profile</th>"
        "<th>discovered</th><th>tx</th><th>rx ok</th><th>rx fail</th>"
        "<th>coll</th><th>low snr</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")

    # discovery matrix: row a, col b, filled if a discovered b
    head = "".join(f"<th>{b}</th>" for b in ids)
    mrows = []
    for a in ids:
        cells = []
        for b in ids:
            if a == b:
                cells.append("<td class='self'>·</td>")
            elif b in s["disc_matrix"][a]:
                cells.append("<td class='hit'></td>")
            else:
                cells.append("<td class='miss'></td>")
        mrows.append(f"<tr><th>{a}</th>{''.join(cells)}</tr>")
    matrix = (f"<table class='mx'><thead><tr><th>a\\b</th>{head}</tr></thead>"
              f"<tbody>{''.join(mrows)}</tbody></table>")

    snr = ("—" if s["snr_mean"] is None else
           f"{s['snr_min']:.1f} / {s['snr_mean']:.1f} / {s['snr_max']:.1f} dB")
    ranges = "".join(
        f"<li>{r['a']}↔{r['b']}: "
        f"{'—' if r['distance_m'] is None else f'{r['distance_m']:.2f} m'}</li>"
        for r in s["ranges"]) or "<li>none</li>"

    return f"""
    <div class="receipt">
      <h1>Mesh run receipt</h1>
      <div class="kv">
        <span><b>run</b> {html.escape(str(meta.get('run_id','?')))}</span>
        <span><b>tier</b> {html.escape(str(meta.get('tier','?')))}</span>
        <span><b>nodes</b> {len(ids)}</span>
        <span><b>duration</b> {s['t_end']:.1f} s</span>
        <span><b>events</b> {s['n_events']}</span>
        <span><b>full discovery</b> {len(s['full_discovery'])}/{len(ids)}</span>
        <span><b>collision rate</b> {s['collision_rate']*100:.1f}%</span>
        <span><b>SINR min/mean/max</b> {snr}</span>
      </div>
      <div class="cols">
        <div><h2>Per-node</h2>{per_table}</div>
        <div><h2>Discovery matrix <small>(row discovered col)</small></h2>{matrix}</div>
        <div><h2>Ranging</h2><ul class="rg">{ranges}</ul></div>
      </div>
    </div>"""


def render(meta: dict, records: list[dict], summary: dict | None = None) -> str:
    s = summary or build_summary(meta, records)
    payload = {
        "meta": meta,
        "events": records,
        "nodes": {str(k): v for k, v in s["nodes"].items()},
        "ids": s["ids"],
        "t_end": s["t_end"],
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    receipt = _receipt_html(meta, s)
    return _PAGE.replace("/*DATA*/", data_json).replace("<!--RECEIPT-->", receipt)


# The page: inline CSS + vanilla-JS canvas piano roll.  __DATA__ is replaced with
# the embedded event payload.  Time is the vertical axis; nodes are lanes.
_PAGE = r"""<meta charset="utf-8">
<title>Mesh piano roll</title>
<style>
  :root { color-scheme: light dark;
    --bg:#0d1117; --fg:#e6edf3; --grid:#21262d; --lane:#161b22; --muted:#8b949e;
    --ok:#3fb950; --fail:#f85149; --defer:#6e7681; --disc:#58a6ff; --ack:#bc8cff;
    --range:#39c5cf; --tx:#8b949e; }
  @media (prefers-color-scheme: light) { :root {
    --bg:#ffffff; --fg:#1f2328; --grid:#e6e8eb; --lane:#f6f8fa; --muted:#59636e; } }
  :root[data-theme=light] { --bg:#fff; --fg:#1f2328; --grid:#e6e8eb; --lane:#f6f8fa; --muted:#59636e; }
  :root[data-theme=dark]  { --bg:#0d1117; --fg:#e6edf3; --grid:#21262d; --lane:#161b22; --muted:#8b949e; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
  .receipt { padding:16px 20px; border-bottom:1px solid var(--grid); }
  .receipt h1 { font-size:18px; margin:0 0 8px; }
  .receipt h2 { font-size:13px; margin:0 0 6px; color:var(--muted); font-weight:600; }
  .receipt h2 small { font-weight:400; }
  .kv { display:flex; flex-wrap:wrap; gap:14px; margin-bottom:14px; }
  .kv span { white-space:nowrap; } .kv b { color:var(--muted); font-weight:600; margin-right:4px; }
  .cols { display:flex; gap:28px; flex-wrap:wrap; align-items:flex-start; }
  table { border-collapse:collapse; }
  .rc td, .rc th { padding:2px 8px; text-align:right; border-bottom:1px solid var(--grid); }
  .rc th { color:var(--muted); font-weight:600; } .rc td:first-child, .rc th:first-child { text-align:left; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
  .mx td, .mx th { width:15px; height:15px; text-align:center; font-size:10px;
    border:1px solid var(--grid); color:var(--muted); }
  .mx td.hit { background:var(--disc); } .mx td.miss { background:transparent; }
  .mx td.self { color:var(--muted); }
  .rg { margin:0; padding-left:18px; } .rg li { list-style:none; }
  .toolbar { position:sticky; top:0; z-index:5; background:var(--bg);
    padding:8px 20px; border-bottom:1px solid var(--grid); display:flex;
    gap:16px; flex-wrap:wrap; align-items:center; }
  .toolbar label { display:inline-flex; align-items:center; gap:5px; cursor:pointer; user-select:none; }
  .sw { width:11px; height:11px; border-radius:2px; display:inline-block; }
  .wrap { display:flex; }
  #legend { color:var(--muted); }
  #tip { position:fixed; pointer-events:none; background:var(--lane);
    border:1px solid var(--grid); border-radius:6px; padding:6px 8px;
    font-size:12px; display:none; max-width:280px; box-shadow:0 4px 14px rgba(0,0,0,.4); z-index:10; }
  canvas { display:block; }
</style>
<!--RECEIPT-->
<div class="toolbar">
  <span id="legend">Legend:</span>
  <label><span class="sw" style="background:var(--tx)"></span>tx</label>
  <label><input type="checkbox" data-k="rx_ok" checked><span class="sw" style="background:var(--ok)"></span>rx ok</label>
  <label><input type="checkbox" data-k="rx_fail" checked><span class="sw" style="background:var(--fail)"></span>rx fail</label>
  <label><input type="checkbox" data-k="discovered" checked><span class="sw" style="background:var(--disc)"></span>discovered</label>
  <label><input type="checkbox" data-k="ack" checked><span class="sw" style="background:var(--ack)"></span>ack</label>
  <label><input type="checkbox" data-k="ranged" checked><span class="sw" style="background:var(--range)"></span>ranged</label>
  <label><input type="checkbox" data-k="carrier_defer"><span class="sw" style="background:var(--defer)"></span>defer</label>
  <label style="margin-left:auto">px/sec <input id="zoom" type="range" min="4" max="60" value="12" style="vertical-align:middle"></label>
</div>
<div class="wrap"><canvas id="roll"></canvas></div>
<div id="tip"></div>
<script>
const DATA = /*DATA*/;
const ids = DATA.ids, nodeInfo = DATA.nodes, events = DATA.events;
const PROF = {clean:'#3fb950', noisy:'#d29922', very_noisy:'#f85149'};
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const laneOf = {}; ids.forEach((n,i)=>laneOf[n]=i);

const cv = document.getElementById('roll'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
const PAD_L = 56, PAD_T = 26, LANE_W = 74, TX_W = 12;
let pxPerSec = 12, hit = [];

const show = {rx_ok:true, rx_fail:true, discovered:true, ack:true, ranged:true, carrier_defer:false};
document.querySelectorAll('.toolbar input[type=checkbox]').forEach(cb=>{
  cb.addEventListener('change',()=>{ show[cb.dataset.k]=cb.checked; draw(); });
});
document.getElementById('zoom').addEventListener('input',e=>{ pxPerSec=+e.target.value; draw(); });

const xLane = i => PAD_L + i*LANE_W + LANE_W/2;
const yTime = t => PAD_T + t*pxPerSec;

function draw(){
  const W = PAD_L + ids.length*LANE_W + 20;
  const H = PAD_T + DATA.t_end*pxPerSec + 40;
  const dpr = window.devicePixelRatio||1;
  cv.width = W*dpr; cv.height = H*dpr; cv.style.width=W+'px'; cv.style.height=H+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  hit = [];

  // lane backgrounds + headers
  ctx.textAlign='center'; ctx.font='11px sans-serif';
  ids.forEach((n,i)=>{
    const x = PAD_L + i*LANE_W;
    ctx.fillStyle = css('--lane'); ctx.globalAlpha = i%2?0.5:0.25;
    ctx.fillRect(x,PAD_T,LANE_W,H-PAD_T-40); ctx.globalAlpha=1;
    const prof = (nodeInfo[n]||{}).profile||'clean';
    ctx.fillStyle = PROF[prof]||'#888';
    ctx.beginPath(); ctx.arc(xLane(i),12,4,0,7); ctx.fill();
    ctx.fillStyle = css('--fg'); ctx.fillText('n'+n, xLane(i)+9, 15);
  });
  // time gridlines every 10s
  ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted');
  ctx.textAlign='right'; ctx.lineWidth=1;
  for(let t=0;t<=DATA.t_end;t+=10){
    const y=yTime(t); ctx.beginPath(); ctx.moveTo(PAD_L,y); ctx.lineTo(W-20,y); ctx.stroke();
    ctx.fillText(t+'s', PAD_L-6, y+3);
  }

  // tx blocks first (so arrows sit on top)
  for(const e of events){
    if(e.kind!=='tx_start') continue;
    const i=laneOf[e.node]; if(i===undefined) continue;
    const y=yTime(e.t), h=Math.max(2,(e.dur||0)*pxPerSec);
    ctx.fillStyle = css('--tx'); ctx.globalAlpha=0.5;
    ctx.fillRect(xLane(i)-TX_W/2, y, TX_W, h); ctx.globalAlpha=1;
    ctx.strokeStyle = css('--tx'); ctx.strokeRect(xLane(i)-TX_W/2, y, TX_W, h);
    hit.push({x:xLane(i), y:y+h/2, r:8, e});
  }

  // arrows / marks
  for(const e of events){
    const i=laneOf[e.node]; if(i===undefined) continue;
    const y=yTime(e.t), x=xLane(i);
    if(e.kind==='rx_ok' && show.rx_ok){
      arrow(e.peer, e.node, y, css('--ok'), 0.5, false); mark(x,y,css('--ok'),e);
    } else if(e.kind==='rx_fail' && show.rx_fail){
      if(e.peer!==undefined) arrow(e.peer, e.node, y, css('--fail'), 0.45, true);
      xmark(x,y,css('--fail'),e);
    } else if(e.kind==='discovered' && show.discovered){
      diamond(x,y,css('--disc'),e);
    } else if(e.kind==='ack' && show.ack){
      mark(x,y,css('--ack'),e);
    } else if(e.kind==='ranged' && show.ranged){
      diamond(x,y,css('--range'),e);
    } else if(e.kind==='carrier_defer' && show.carrier_defer){
      tick(x,y,css('--defer'),e);
    }
  }
}

function arrow(from,to,y,color,alpha,dashed){
  if(from===undefined||to===undefined) return;
  const i=laneOf[from], j=laneOf[to]; if(i===undefined||j===undefined||i===j) return;
  const x0=xLane(i), x1=xLane(j);
  ctx.save(); ctx.globalAlpha=alpha; ctx.strokeStyle=color; ctx.fillStyle=color;
  ctx.lineWidth=1; if(dashed) ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  const dir=Math.sign(x1-x0)||1;
  ctx.beginPath(); ctx.moveTo(x1,y); ctx.lineTo(x1-6*dir,y-3); ctx.lineTo(x1-6*dir,y+3);
  ctx.closePath(); ctx.fill(); ctx.restore();
}
function mark(x,y,c,e){ ctx.fillStyle=c; ctx.beginPath(); ctx.arc(x,y,3,0,7); ctx.fill(); hit.push({x,y,r:5,e}); }
function xmark(x,y,c,e){ ctx.strokeStyle=c; ctx.lineWidth=1.4; ctx.beginPath();
  ctx.moveTo(x-3,y-3); ctx.lineTo(x+3,y+3); ctx.moveTo(x+3,y-3); ctx.lineTo(x-3,y+3); ctx.stroke();
  hit.push({x,y,r:5,e}); }
function diamond(x,y,c,e){ ctx.fillStyle=c; ctx.beginPath();
  ctx.moveTo(x,y-4); ctx.lineTo(x+4,y); ctx.lineTo(x,y+4); ctx.lineTo(x-4,y); ctx.closePath(); ctx.fill();
  hit.push({x,y,r:6,e}); }
function tick(x,y,c,e){ ctx.strokeStyle=c; ctx.lineWidth=1; ctx.strokeRect(x-3,y-1,6,2); hit.push({x,y,r:5,e}); }

// hover tooltip via nearest-hit test
cv.addEventListener('mousemove', ev=>{
  const rect=cv.getBoundingClientRect(); const mx=ev.clientX-rect.left, my=ev.clientY-rect.top;
  let best=null, bd=1e9;
  for(const h of hit){ const d=(h.x-mx)**2+(h.y-my)**2; if(d<h.r*h.r && d<bd){bd=d;best=h;} }
  if(best){
    const e=best.e; const f=[];
    f.push('<b>'+e.kind+'</b> @ t='+e.t+'s');
    if(e.node!==undefined) f.push('node '+e.node+(e.t_local!==undefined?' (local '+e.t_local+'s)':''));
    if(e.peer!==undefined) f.push('peer '+e.peer);
    if(e.ptype) f.push('type '+e.ptype+(e.pseq!==undefined?' #'+e.pseq:''));
    if(e.snr_db!==undefined) f.push('SINR '+e.snr_db+' dB');
    if(e.reason) f.push('reason '+e.reason);
    if(e.dur!==undefined) f.push('airtime '+e.dur+'s');
    if(e.detail) f.push(JSON.stringify(e.detail));
    tip.innerHTML=f.join('<br>');
    tip.style.display='block'; tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY+12)+'px';
  } else tip.style.display='none';
});
cv.addEventListener('mouseleave',()=>tip.style.display='none');
draw();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a mesh events.jsonl piano roll.")
    ap.add_argument("events", help="path to events.jsonl")
    ap.add_argument("-o", "--out", help="output HTML path (default: alongside input)")
    args = ap.parse_args()

    meta, records = ev.load_events(args.events)
    if not records:
        print(f"no events in {args.events}", file=sys.stderr)
        return 1
    summary = build_summary(meta, records)
    out = args.out or (os.path.splitext(args.events)[0] + ".html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(meta, records, summary))
    print(f"wrote {out}  ({summary['n_events']} events, {len(summary['ids'])} nodes, "
          f"{len(summary['full_discovery'])}/{len(summary['ids'])} full discovery)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
