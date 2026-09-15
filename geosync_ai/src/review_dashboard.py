"""Generate a self-contained, offline review dashboard for the demo output.

The page is deliberately read-only: it helps judges and reviewers inspect the
pipeline's evidence without pretending that a browser button can approve an
official land-record change.
"""
from __future__ import annotations

import json
from pathlib import Path


def _read_geojson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_dashboard(output_dir: Path, summary: dict, review_queue: dict) -> Path:
    """Write an offline HTML evidence viewer beside the generated outputs."""
    geojson = _read_geojson(output_dir / "harmonized_output.geojson")
    page_data = json.dumps({"summary": summary, "queue": review_queue, "geojson": geojson})
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GeoSync AI | Review Console</title>
<style>
:root { --ink:#102a43; --muted:#627d98; --paper:#f4f7fb; --card:#fff; --line:#d9e2ec; --green:#16803c; --amber:#b45309; --red:#b42318; --blue:#175cd3; }
* { box-sizing:border-box; } body { margin:0; font:14px/1.45 Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--paper); }
header { padding:28px max(24px,calc((100vw - 1180px)/2)); background:linear-gradient(115deg,#09203f,#175cd3); color:#fff; }
header h1 { margin:0 0 5px; font-size:26px; } header p { margin:0; color:#dbeafe; max-width:850px; }
main { max-width:1180px; margin:24px auto 48px; padding:0 24px; } .notice { background:#fff8e6; border:1px solid #f6d78b; padding:12px 14px; border-radius:9px; color:#714b00; margin-bottom:18px; }
.cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:18px; } .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:15px; box-shadow:0 1px 2px #102a4309; } .value { font-size:25px; font-weight:750; } .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.grid { display:grid; grid-template-columns:1.25fr .75fr; gap:18px; } section { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; } section h2 { font-size:16px; margin:0; padding:14px 16px; border-bottom:1px solid var(--line); } .tools { display:flex; gap:8px; padding:12px 16px; border-bottom:1px solid var(--line); } button { border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 9px; color:var(--ink); cursor:pointer; } button.active { background:#e7f0ff; color:var(--blue); border-color:#91b8f7; }
table { width:100%; border-collapse:collapse; } th,td { text-align:left; padding:10px 12px; border-bottom:1px solid #eef2f6; } th { font-size:11px; color:var(--muted); text-transform:uppercase; } tbody tr { cursor:pointer; } tbody tr:hover,tbody tr.selected { background:#f0f6ff; }.badge { display:inline-block; border-radius:999px; padding:2px 7px; font-size:12px; font-weight:650; }.high { background:#e7f7ed;color:var(--green); }.medium { background:#fff3df;color:var(--amber); }.low { background:#fee9e7;color:var(--red); }
#detail { padding:16px; min-height:270px; } .empty { color:var(--muted); } dl { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:16px 0; } dt { color:var(--muted); font-size:12px; } dd { margin:2px 0 0; font-weight:650; } #map { width:100%; height:250px; display:block; background:#edf4ee; border-top:1px solid var(--line); } .map-feature { stroke:#102a43; stroke-width:.12; fill:#4caf6c; fill-opacity:.56; } .map-feature.manual_review { fill:#d92d20; } .map-feature.proposed_flagged { fill:#f79009; }
.footer { color:var(--muted); font-size:12px; margin:16px 0; } @media(max-width:850px){ .cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr} } @media(max-width:480px){.cards{grid-template-columns:1fr}}
</style>
</head>
<body><header><h1>GeoSync AI Review Console</h1><p>Evidence-led review of synthetic prototype proposals across imagery and government-record layers.</p></header>
<main><div class="notice"><strong>Prototype safeguard:</strong> this offline console is read-only. It displays AI-assisted proposals, not legally valid land-record decisions. A production deployment requires authenticated roles, protected audit storage, and authorised approval workflows.</div>
<div class="cards" id="cards"></div><div class="grid"><section><h2>Proposed match queue</h2><div class="tools"><button class="active" data-filter="all">All cases</button><button data-filter="manual_review_required">Manual review</button><button data-filter="eligible_for_authorized_workflow">High-confidence proposals</button></div><table><thead><tr><th>Case</th><th>Confidence</th><th>Route</th><th>Support</th></tr></thead><tbody id="cases"></tbody></table></section><section><h2>Evidence detail</h2><div id="detail" class="empty">Select a proposed match to inspect its evidence.</div><svg id="map" viewBox="0 0 100 100" aria-label="Harmonized footprint overview"></svg></section></div>
<p class="footer">Generated locally from <code>harmonized_output.geojson</code>, <code>review_queue.json</code>, and <code>run_summary.json</code>. Artifact hashes and a local prototype audit event are available in this output directory.</p></main>
<script>
const data = __DATA__;
const cases = data.queue.cases;
let filter = 'all', selected = null;
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function badge(tier){ return `<span class="badge ${tier}">${esc(tier)}</span>`; }
function renderCards(){ const s=data.summary, manual=cases.filter(c=>c.prototype_route==='manual_review_required').length; document.querySelector('#cards').innerHTML = [ ['Extracted features',s.n_extracted_features], ['Synthetic adaptive IoU',s.adaptive_pipeline_iou], ['Manual-review cases',manual], ['Quality gate',s.data_quality_gate] ].map(([l,v])=>`<div class="card"><div class="value">${esc(v)}</div><div class="label">${esc(l)}</div></div>`).join(''); }
function renderCases(){ const view=cases.filter(c=>filter==='all'||c.prototype_route===filter); document.querySelector('#cases').innerHTML=view.map(c=>`<tr data-id="${esc(c.case_id)}" class="${selected===c.case_id?'selected':''}"><td><strong>${esc(c.case_id)}</strong><br>${badge(c.confidence_tier)}</td><td>${esc(c.confidence)}</td><td>${esc(c.prototype_route.replaceAll('_',' '))}</td><td>${esc(c.evidence.cross_source_support_count)} sources</td></tr>`).join('') || '<tr><td colspan="4" class="empty">No cases in this view.</td></tr>'; document.querySelectorAll('#cases tr[data-id]').forEach(row=>row.onclick=()=>{selected=row.dataset.id;renderCases();renderDetail();}); }
function renderDetail(){ const el=document.querySelector('#detail'), c=cases.find(x=>x.case_id===selected); if(!c){el.className='empty';el.textContent='Select a proposed match to inspect its evidence.';return;} el.className=''; const e=c.evidence; el.innerHTML=`<strong>${esc(c.case_id)}</strong> ${badge(c.confidence_tier)}<p>${esc(c.prototype_route.replaceAll('_',' '))}. Final decision: <strong>pending authorised workflow</strong>.</p><dl><div><dt>Spatial overlap (IoU)</dt><dd>${esc(e.iou)}</dd></div><div><dt>Centroid distance</dt><dd>${esc(e.centroid_distance_m)} m</dd></div><div><dt>Area ratio</dt><dd>${esc(e.area_ratio)}</dd></div><div><dt>Attribute score</dt><dd>${esc(e.attribute_score)}</dd></div><div><dt>Cross-source support</dt><dd>${esc(e.cross_source_support_count)} sources</dd></div><div><dt>Extraction</dt><dd>${esc(e.extraction_method)} (${esc(e.extraction_confidence)})</dd></div></dl><p class="footer">Permitted review actions are enforced only by the future authenticated service; this static demo cannot approve, reject, or alter a record.</p>`; }
function renderMap(){ const features=data.geojson.features||[], svg=document.querySelector('#map'), coords=[]; features.forEach(f=>(f.geometry?.coordinates?.[0]||[]).forEach(p=>coords.push(p))); if(!coords.length)return; const xs=coords.map(p=>p[0]),ys=coords.map(p=>p[1]),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),dx=maxx-minx||1,dy=maxy-miny||1; svg.innerHTML=features.map(f=>{const ring=f.geometry?.coordinates?.[0]||[], points=ring.map(([x,y])=>`${4+92*(x-minx)/dx},${96-92*(y-miny)/dy}`).join(' ');return `<polygon class="map-feature ${esc(f.properties?.status||'')}" points="${points}"/>`;}).join(''); }
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===button));renderCases();});renderCards();renderCases();renderDetail();renderMap();
</script></body></html>"""
    dashboard_path = output_dir / "review_dashboard.html"
    dashboard_path.write_text(html.replace("__DATA__", page_data), encoding="utf-8")
    return dashboard_path
