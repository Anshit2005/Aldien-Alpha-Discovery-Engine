"""Create a self-contained HTML dashboard from a paired search comparison JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON input {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("The input JSON must contain an object at the top level")
    return payload


def _details_for_run(summary_path: Path, run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    files = run.get("result_files", {})
    details: dict[str, list[dict[str, Any]]] = {}
    for method in ("evolutionary", "random"):
        raw_path = files.get(method)
        if not raw_path:
            continue
        candidate_path = Path(raw_path)
        if not candidate_path.is_file():
            candidate_path = summary_path.parent / candidate_path.name
        if not candidate_path.is_file():
            continue
        try:
            result = _load_json(candidate_path)
        except ValueError:
            continue
        validation = {
            item.get("expression"): item.get("fitness", {})
            for item in result.get("validation", [])
            if isinstance(item, dict) and "fitness" in item
        }
        candidates = []
        for item in result.get("best", []):
            if not isinstance(item, dict):
                continue
            fitness = item.get("fitness", {})
            validation_fitness = validation.get(item.get("expression"), {})
            candidates.append({
                "expression": item.get("expression", ""),
                "train_score": item.get("score"),
                "train_icir": fitness.get("icir"),
                "train_mean_ic": fitness.get("mean_ic"),
                "validation_icir": validation_fitness.get("icir"),
                "validation_mean_ic": validation_fitness.get("mean_ic"),
                "validation_coverage": validation_fitness.get("coverage"),
            })
        details[method] = candidates
    return details


def _prepare_view_data(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("paired_runs")
    if not isinstance(runs, list):
        raise ValueError(
            "Expected a paired comparison JSON containing a 'paired_runs' array. "
            "Use the comparison_summary.json produced by run_comparison.py."
        )
    prepared_runs = []
    for raw_run in runs:
        if not isinstance(raw_run, dict):
            continue
        run = dict(raw_run)
        run["candidate_details"] = _details_for_run(summary_path, run)
        prepared_runs.append(run)
    return {
        "dataset": payload.get("dataset", {}),
        "protocol": payload.get("protocol", {}),
        "paired_runs": prepared_runs,
    }


def _embedded_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")


def render_html(view_data: dict[str, Any], title: str | None = None) -> str:
    dataset = view_data.get("dataset", {})
    protocol = view_data.get("protocol", {})
    page_title = title or "Aldien · Search comparison"
    escaped_title = html.escape(page_title)
    data_json = _embedded_json(view_data)
    source_name = html.escape(str(dataset.get("path", "Research dataset")))
    start_date = html.escape(str(dataset.get("date_min", "—")))
    end_date = html.escape(str(dataset.get("date_max", "—")))
    horizon = html.escape(str(protocol.get("evolution_config", {}).get("horizon", "—")))
    top_k = html.escape(str(protocol.get("top_k", "—")))
    fingerprint = html.escape(str(dataset.get("sha256", ""))[:12])
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{escaped_title}</title>
<style>
:root{{--ink:#162b35;--muted:#60757e;--paper:#f2f5f2;--card:#fff;--line:#dce5e2;--green:#087e72;--mint:#cbeee3;--orange:#d77a3e;--blue:#4777a7;--red:#b94e51;--shadow:0 12px 34px rgba(30,59,66,.07)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1440px;margin:0 auto;padding:30px 32px 64px}}header{{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:24px}}
.eyebrow{{color:var(--green);font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}}h1{{font-size:clamp(28px,4vw,44px);letter-spacing:-.045em;line-height:1.1;margin:8px 0 12px}}.sub{{color:var(--muted);max-width:850px;margin:0}}
.badge{{background:var(--ink);color:#f4fbf8;padding:10px 14px;border-radius:999px;font-size:12px;font-weight:700;white-space:nowrap}}
.meta{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 26px}}.chip{{border:1px solid var(--line);background:#ffffffaa;border-radius:999px;padding:6px 11px;color:#46606a;font-size:12px}}.chip strong{{color:var(--ink)}}
.section-title{{display:flex;justify-content:space-between;align-items:end;margin:30px 2px 12px;gap:12px}}h2{{font-size:18px;letter-spacing:-.02em;margin:0}}.hint{{font-size:12px;color:var(--muted)}}
.grid{{display:grid;gap:14px}}.kpis{{grid-template-columns:repeat(5,minmax(0,1fr))}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}}.kpi{{padding:18px 20px;min-height:118px}}.kpi-label{{color:var(--muted);font-size:12px;font-weight:700}}.kpi-value{{font-size:30px;letter-spacing:-.04em;font-weight:750;margin:9px 0 2px}}.kpi-note{{font-size:11px;color:var(--muted)}}
.charts{{grid-template-columns:1.25fr 1fr 1fr}}.chart-card{{padding:18px 18px 12px;min-width:0}}.chart-title{{font-weight:750;font-size:14px;margin-bottom:3px}}.chart-note{{font-size:11px;color:var(--muted);min-height:32px}}canvas{{width:100%;height:250px;display:block}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:960px}}th,td{{padding:12px 13px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:#f8faf9}}th:first-child,td:first-child{{text-align:left}}tbody tr:last-child td{{border-bottom:0}}.winner{{color:var(--green);font-weight:800}}.negative{{color:var(--red)}}
.method-grid{{grid-template-columns:1fr 1fr}}.method-card{{padding:20px}}.method-head{{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}}.method-name{{font-size:16px;font-weight:800}}.method-dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:8px;background:var(--green)}}.random-dot{{background:var(--orange)}}.method-stats{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.stat{{padding:12px;border-radius:12px;background:#f5f8f6}}.stat span{{display:block;color:var(--muted);font-size:11px}}.stat strong{{font-size:18px}}
details{{border-top:1px solid var(--line);padding-top:12px;margin-top:16px}}summary{{cursor:pointer;color:var(--green);font-size:12px;font-weight:750}}.formula-list{{display:grid;gap:9px;margin-top:12px}}.formula{{padding:12px;background:#f7faf8;border:1px solid var(--line);border-radius:11px}}.formula-top{{display:flex;justify-content:space-between;gap:10px;margin-bottom:6px;font-size:11px;color:var(--muted)}}code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere;color:#173c46}}
.footnote{{padding:16px 18px;color:var(--muted);font-size:12px;margin-top:18px}}.empty{{color:var(--muted);padding:20px}}.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:5px}}.legend i{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}}
@media(max-width:1050px){{.kpis{{grid-template-columns:repeat(3,1fr)}}.charts{{grid-template-columns:1fr 1fr}}.charts .chart-card:first-child{{grid-column:span 2}}}}
@media(max-width:680px){{.wrap{{padding:22px 14px 40px}}header{{display:block}}.badge{{display:inline-block;margin-top:16px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.charts,.method-grid{{grid-template-columns:1fr}}.charts .chart-card:first-child{{grid-column:auto}}.kpi-value{{font-size:25px}}}}
@media print{{body{{background:#fff}}.wrap{{max-width:none;padding:0}}.card{{box-shadow:none;break-inside:avoid}}details{{display:none}}}}
</style>
</head>
<body><main class="wrap">
<header><div><div class="eyebrow">Aldien · Alpha Discovery Engine</div><h1>Search experiment report</h1><p class="sub">A paired view of random and evolutionary alpha search. Both methods use the same market data, seed pool, train objective, and per-seed evaluation budget.</p></div><div class="badge">RESEARCH RUN · VALIDATION ONLY</div></header>
<div class="meta"><span class="chip"><strong>Dataset</strong> {source_name}</span><span class="chip"><strong>Dates</strong> {start_date} → {end_date}</span><span class="chip"><strong>Horizon</strong> {horizon}d</span><span class="chip"><strong>Top K</strong> {top_k}</span><span class="chip"><strong>Fingerprint</strong> {fingerprint or 'not recorded'}</span></div>
<div class="section-title"><h2>At a glance</h2><span class="hint" id="run-summary"></span></div><section class="grid kpis" id="kpis"></section>
<div class="section-title"><h2>Across-seed comparison</h2><span class="hint">Hover is not required; values are printed in the charts and table.</span></div>
<section class="grid charts">
 <article class="card chart-card"><div class="chart-title">Validation ICIR by seed</div><div class="chart-note">Mean signed validation ICIR for each method’s top-K train-ranked formulas.</div><canvas id="validation-chart"></canvas><div class="legend"><span><i style="background:#087e72"></i>Evolutionary</span><span><i style="background:#d77a3e"></i>Random</span></div></article>
 <article class="card chart-card"><div class="chart-title">Mean method performance</div><div class="chart-note">Average top-K training |ICIR| and signed validation ICIR across paired seeds.</div><canvas id="performance-chart"></canvas></article>
 <article class="card chart-card"><div class="chart-title">Runtime per seed</div><div class="chart-note">Elapsed seconds for each complete search run, including finalist validation.</div><canvas id="runtime-chart"></canvas><div class="legend"><span><i style="background:#087e72"></i>Evolutionary</span><span><i style="background:#d77a3e"></i>Random</span></div></article>
</section>
<div class="section-title"><h2>Per-seed metrics</h2><span class="hint">Validation is reported for candidates selected on training data.</span></div><section class="card table-wrap"><table><thead><tr><th>Seed</th><th>Budget</th><th>Best train |ICIR|<br>Evolution</th><th>Best train |ICIR|<br>Random</th><th>Mean val ICIR<br>Evolution</th><th>Mean val ICIR<br>Random</th><th>Train→validation<br>degradation · Evo</th><th>Direction<br>agreement · Evo</th><th>Runtime · Evo / Random</th><th>Val winner</th></tr></thead><tbody id="runs-table"></tbody></table><div id="empty-runs" class="empty" hidden>No paired runs were found in this JSON.</div></section>
<div class="section-title"><h2>Method summaries</h2><span class="hint">Averages across available paired runs; not a substitute for significance testing.</span></div><section class="grid method-grid" id="method-summaries"></section>
<div class="section-title"><h2>Top formulas by seed</h2><span class="hint">Candidate details are loaded from the paired result files when available.</span></div><section id="formula-sections" class="grid"></section>
<section class="card footnote" id="protocol-note"></section>
</main>
<script id="report-data" type="application/json">{data_json}</script>
<script>
const D=JSON.parse(document.getElementById('report-data').textContent), C={{evolutionary:'#087e72',random:'#d77a3e',grid:'#e4ece9',ink:'#29434c',muted:'#71848a',blue:'#4777a7'}};
const runs=D.paired_runs||[], methods=['evolutionary','random'], labels={{evolutionary:'Evolutionary search',random:'Random search'}};
const num=(x,d=3)=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':Number(x).toFixed(d);
const signed=(x,d=3)=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':(Number(x)>0?'+':'')+Number(x).toFixed(d);
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
const vals=(m,k)=>runs.map(r=>r[m]?.[k]).filter(x=>x!==null&&x!==undefined&&Number.isFinite(Number(x))).map(Number);
const fmtInt=x=>x===null||x===undefined?'—':Number(x).toLocaleString();
function addKpi(label,value,note){{const el=document.createElement('article');el.className='card kpi';for(const [cls,txt] of [['kpi-label',label],['kpi-value',value],['kpi-note',note]]){{const d=document.createElement('div');d.className=cls;d.textContent=txt;el.appendChild(d)}}document.getElementById('kpis').appendChild(el)}}
function init(){{
 const evoWins=runs.filter(r=>Number(r.evolutionary?.mean_validation_icir_top_k)>Number(r.random?.mean_validation_icir_top_k)).length;
 const evals=mean(runs.map(r=>Number(r.evaluation_budget)||0));
 const evoVal=mean(vals('evolutionary','mean_validation_icir_top_k')),randVal=mean(vals('random','mean_validation_icir_top_k'));
 addKpi('Paired seeds',String(runs.length),'Each seed runs both methods');
 addKpi('Mean validation ICIR · evolution',signed(evoVal),'Top-K formulas, signed');
 addKpi('Mean validation ICIR · random',signed(randVal),'Top-K formulas, signed');
 addKpi('Evolution validation wins',runs.length?`${{evoWins}} / ${{runs.length}}`:'—','By per-seed mean validation ICIR');
 addKpi('Mean matched budget',fmtInt(evals),'Unique train evaluations per method');
 document.getElementById('run-summary').textContent=`${{runs.length}} paired run${{runs.length===1?'':'s'}} · ${{fmtInt(D.dataset?.rows)}} dataset rows`;
 renderTable();renderSummaries();renderFormulas();renderProtocol();
 drawLine(document.getElementById('validation-chart'),runs.map(r=>String(r.seed)),[
  {{name:'Evolutionary',color:C.evolutionary,values:runs.map(r=>r.evolutionary?.mean_validation_icir_top_k)}},
  {{name:'Random',color:C.random,values:runs.map(r=>r.random?.mean_validation_icir_top_k)}}]);
 drawBars(document.getElementById('performance-chart'),[
  {{label:'Train |ICIR|',evo:mean(vals('evolutionary','mean_train_abs_icir_top_k')),random:mean(vals('random','mean_train_abs_icir_top_k'))}},
  {{label:'Validation ICIR',evo:evoVal,random:randVal}}]);
 drawBars(document.getElementById('runtime-chart'),runs.map(r=>({{label:String(r.seed),evo:r.evolutionary?.elapsed_seconds,random:r.random?.elapsed_seconds}})),true);
}}
function renderTable(){{const body=document.getElementById('runs-table');if(!runs.length)document.getElementById('empty-runs').hidden=false;
 for(const r of runs){{const e=r.evolutionary||{{}},q=r.random||{{}},winner=Number(e.mean_validation_icir_top_k)>Number(q.mean_validation_icir_top_k)?'Evolution':'Random';const tr=document.createElement('tr');const cells=[r.seed,fmtInt(r.evaluation_budget),num(e.best_train_abs_icir),num(q.best_train_abs_icir),signed(e.mean_validation_icir_top_k),signed(q.mean_validation_icir_top_k),num(e.train_to_validation_abs_icir_degradation),e.train_validation_direction_agreement_rate==null?'—':`${{(100*e.train_validation_direction_agreement_rate).toFixed(0)}}%`,`${{num(e.elapsed_seconds,1)}}s / ${{num(q.elapsed_seconds,1)}}s`,winner];cells.forEach((v,i)=>{{const td=document.createElement('td');td.textContent=v;if((i===4&&Number(e.mean_validation_icir_top_k)>Number(q.mean_validation_icir_top_k))||(i===5&&Number(q.mean_validation_icir_top_k)>Number(e.mean_validation_icir_top_k)))td.className='winner';if((i===4||i===5)&&String(v).startsWith('-'))td.classList.add('negative');tr.appendChild(td)}});body.appendChild(tr)}}}}
function renderSummaries(){{const root=document.getElementById('method-summaries');for(const m of methods){{const card=document.createElement('article');card.className='card method-card';const head=document.createElement('div');head.className='method-head';const name=document.createElement('div');name.className='method-name';const dot=document.createElement('i');dot.className=m==='random'?'method-dot random-dot':'method-dot';name.appendChild(dot);name.append(labels[m]);head.appendChild(name);card.appendChild(head);const stats=document.createElement('div');stats.className='method-stats';const diversity=mean(runs.map(r=>r[m]?.top_k_expression_diversity?.mean_pairwise_operator_jaccard_distance).filter(x=>x!=null&&Number.isFinite(Number(x))).map(Number));const items=[['Best train |ICIR|',mean(vals(m,'best_train_abs_icir'))],['Mean top-K train |ICIR|',mean(vals(m,'mean_train_abs_icir_top_k'))],['Mean top-K validation ICIR',mean(vals(m,'mean_validation_icir_top_k'))],['Mean validation |ICIR|',mean(vals(m,'mean_validation_abs_icir_top_k'))],['Mean run time',mean(vals(m,'elapsed_seconds'))],['Mean expression diversity',diversity]];for(const [label,value] of items){{const box=document.createElement('div');box.className='stat';const a=document.createElement('span');a.textContent=label;const b=document.createElement('strong');b.textContent=label==='Mean run time'?`${{num(value,1)}} s`:signed(value);box.append(a,b);stats.appendChild(box)}}card.appendChild(stats);root.appendChild(card)}}}}
function renderFormulas(){{const root=document.getElementById('formula-sections');let count=0;for(const r of runs)for(const m of methods){{const formulas=r.candidate_details?.[m]||[];if(!formulas.length)continue;count++;const d=document.createElement('details');d.className='card';d.style.padding='15px 18px';const s=document.createElement('summary');s.textContent=`Seed ${{r.seed}} · ${{labels[m]}} · top ${{formulas.length}}`;d.appendChild(s);const list=document.createElement('div');list.className='formula-list';formulas.forEach((f,i)=>{{const item=document.createElement('div');item.className='formula';const top=document.createElement('div');top.className='formula-top';top.textContent=`#${{i+1}} · train |ICIR| ${{num(f.train_score)}} · validation ICIR ${{signed(f.validation_icir)}}`;const code=document.createElement('code');code.textContent=f.expression;item.append(top,code);list.appendChild(item)}});d.appendChild(list);root.appendChild(d)}}if(!count){{const p=document.createElement('div');p.className='card empty';p.textContent='No per-run candidate files were found beside the comparison JSON. Metrics and charts are still available.';root.appendChild(p)}}}}
function renderProtocol(){{const p=D.protocol||{{}},d=D.dataset||{{}},cfg=p.evolution_config||{{}},el=document.getElementById('protocol-note');el.textContent=`Protocol: objective ${{p.fitness_objective||'—'}}; train split ${{p.train_split||'—'}}; validation split ${{p.validation_split||'—'}}; validation used for selection: ${{p.validation_used_for_selection?'yes':'no'}}. Horizon ${{cfg.horizon??'—'}} days; max depth ${{cfg.max_depth??'—'}}; seed pool ${{p.evolution_seed_pool||'—'}}; dataset SHA-256 ${{d.sha256||'not recorded'}}. Train-to-validation degradation is mean top-K training |ICIR| minus mean top-K validation |ICIR|; negative values mean validation magnitude exceeded training magnitude.`}}
function canvasSetup(canvas){{const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(300,rect.width*ratio);canvas.height=250*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);return {{ctx,w:canvas.width/ratio,h:250}}}}
function drawLine(canvas,labels,series){{const {{ctx,w,h}}=canvasSetup(canvas),pad={{l:42,r:14,t:16,b:36}},all=series.flatMap(s=>s.values.map(Number).filter(Number.isFinite));if(!all.length)return;let lo=Math.min(0,...all),hi=Math.max(0,...all);if(hi-lo<.02){{hi+=.01;lo-=.01}}const x=i=>pad.l+(labels.length<2?(w-pad.l-pad.r)/2:i*(w-pad.l-pad.r)/(labels.length-1)),y=v=>pad.t+(hi-v)*(h-pad.t-pad.b)/(hi-lo);ctx.font='11px system-ui';ctx.strokeStyle=C.grid;ctx.fillStyle=C.muted;for(let i=0;i<5;i++){{const v=hi-i*(hi-lo)/4,yy=y(v);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillText(v.toFixed(2),3,yy+4)}}labels.forEach((label,i)=>ctx.fillText(label,x(i)-8,h-10));for(const s of series){{ctx.beginPath();ctx.strokeStyle=s.color;ctx.lineWidth=2.5;let started=false;s.values.forEach((v,i)=>{{if(v==null||!Number.isFinite(Number(v)))return;const xx=x(i),yy=y(Number(v));if(!started){{ctx.moveTo(xx,yy);started=true}}else ctx.lineTo(xx,yy)}});ctx.stroke();s.values.forEach((v,i)=>{{if(v==null||!Number.isFinite(Number(v)))return;ctx.beginPath();ctx.fillStyle=s.color;ctx.arc(x(i),y(Number(v)),4,0,Math.PI*2);ctx.fill()}})}}}}
function drawBars(canvas,items,runtime=false){{const {{ctx,w,h}}=canvasSetup(canvas),pad={{l:44,r:12,t:16,b:40}},vals=items.flatMap(x=>[Number(x.evo),Number(x.random)]).filter(Number.isFinite);if(!vals.length)return;let min=Math.min(0,...vals),max=Math.max(0,...vals);if(max-min<.01){{max+=.01;min-=.01}}const y=v=>pad.t+(max-v)*(h-pad.t-pad.b)/(max-min),base=y(0),groupW=(w-pad.l-pad.r)/Math.max(items.length,1),barW=Math.min(28,groupW*.24);ctx.font='10px system-ui';ctx.strokeStyle=C.grid;ctx.fillStyle=C.muted;for(let i=0;i<5;i++){{const v=max-i*(max-min)/4,yy=y(v);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillText(runtime?Math.round(v)+'s':v.toFixed(2),2,yy+4)}}items.forEach((item,i)=>{{const cx=pad.l+groupW*(i+.5);[[item.evo,C.evolutionary,-1],[item.random,C.random,1]].forEach(([value,color,side])=>{{if(value==null||!Number.isFinite(Number(value)))return;const yy=y(Number(value)),top=Math.min(yy,base),height=Math.max(1,Math.abs(base-yy));ctx.fillStyle=color;ctx.fillRect(cx+side*(barW+3)-barW/2,top,barW,height)}});ctx.fillStyle=C.muted;const label=runtime?'Seed '+item.label:item.label;ctx.fillText(label,cx-ctx.measureText(label).width/2,h-12)}})}}
init();window.addEventListener('resize',()=>{{document.querySelectorAll('canvas').forEach(c=>{{const id=c.id;c.getContext('2d').setTransform(1,0,0,1,0,0);if(id==='validation-chart')drawLine(c,runs.map(r=>String(r.seed)),[{{name:'Evolutionary',color:C.evolutionary,values:runs.map(r=>r.evolutionary?.mean_validation_icir_top_k)}},{{name:'Random',color:C.random,values:runs.map(r=>r.random?.mean_validation_icir_top_k)}}]);if(id==='performance-chart')drawBars(c,[{{label:'Train |ICIR|',evo:mean(vals('evolutionary','mean_train_abs_icir_top_k')),random:mean(vals('random','mean_train_abs_icir_top_k'))}},{{label:'Validation ICIR',evo:mean(vals('evolutionary','mean_validation_icir_top_k')),random:mean(vals('random','mean_validation_icir_top_k'))}}]);if(id==='runtime-chart')drawBars(c,runs.map(r=>({{label:String(r.seed),evo:r.evolutionary?.elapsed_seconds,random:r.random?.elapsed_seconds}})),true)}})}});
</script></body></html>'''


def generate_report(input_path: Path, output_path: Path, title: str | None = None) -> None:
    payload = _load_json(input_path)
    view_data = _prepare_view_data(input_path, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(view_data, title=title), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="comparison_summary.json")
    parser.add_argument("--output", "-o", type=Path, help="HTML output path (defaults beside input)")
    parser.add_argument("--title", help="Optional report heading/title")
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".html")
    try:
        generate_report(args.input, output, title=args.title)
    except ValueError as error:
        parser.error(str(error))
    print(f"HTML report: {output}")


if __name__ == "__main__":
    main()
