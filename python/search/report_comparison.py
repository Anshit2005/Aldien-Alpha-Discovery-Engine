"""
Create a self-contained HTML report from a search comparison summary.

Works for any set of methods in the summary: evolution and random search,
plus LLM-guided evolution and its no-retrieval ablation when they were run.
The report shows Tables 2 and 3 (the same numbers as tables.md), charts per
seed, the LLM operator's counts, and each run's top formulas with their
filter and test results.

    python -m python.search.report_comparison --input data/processed/experiment/comparison_summary.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from python.search.direction import directional_fitness, signal_direction
from python.search.tables import (
    FILTER_TITLE,
    LABELS,
    QUALITY_TITLE,
    filter_rows,
    methods_of,
    notes,
    quality_rows,
)
from python.stats.run_filter import filtered_path

COLORS = {
    "evolutionary": "#087e72",
    "random": "#d77a3e",
    "evolutionary_llm": "#4777a7",
    "evolutionary_llm_no_retrieval": "#8a5fb0",
}
FALLBACK_COLORS = ("#5d6d74", "#b94e51", "#9a7b1c")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON input {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("The input JSON must contain an object at the top level")
    return payload


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _resolve(summary_path: Path, raw_path: str) -> Path | None:
    """A result file as recorded, or by name beside the summary (folder moved)."""

    path = Path(raw_path)
    if path.is_file():
        return path
    path = summary_path.parent / path.name
    return path if path.is_file() else None


def _filter_results(result_file: Path) -> dict[str, dict[str, Any]] | None:
    """Expression -> filter verdict and test result, or None when not filtered."""

    path = filtered_path(result_file)
    if not path.is_file():
        return None
    try:
        filtered = _load_json(path)
    except ValueError:
        return None
    verdicts = {
        item.get("expression"): {"passed": bool(item.get("passed")), "p_value": item.get("p_value")}
        for item in filtered.get("formulas", [])
        if isinstance(item, dict)
    }
    for item in (filtered.get("test") or {}).get("results", []):
        if isinstance(item, dict) and item.get("expression") in verdicts:
            verdicts[item["expression"]].update(
                test_directional_icir=_finite_or_none((item.get("directional") or {}).get("icir")),
                held_on_test=item.get("held"),
            )
    return verdicts


def _details_for_run(summary_path: Path, run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    for method, raw_path in (run.get("result_files") or {}).items():
        candidate_path = _resolve(summary_path, raw_path) if raw_path else None
        if candidate_path is None:
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
        verdicts = _filter_results(candidate_path)
        candidates = []
        for item in result.get("best", []):
            if not isinstance(item, dict):
                continue
            fitness = item.get("fitness", {})
            validation_fitness = validation.get(item.get("expression"), {})
            direction = item.get("direction", signal_direction(fitness.get("icir")))
            oriented = directional_fitness(validation_fitness, direction) if validation_fitness else {}
            verdict = (verdicts or {}).get(item.get("expression"), {})
            candidates.append({
                "expression": item.get("expression", ""),
                "train_score": item.get("score"),
                "train_icir": fitness.get("icir"),
                "train_mean_ic": fitness.get("mean_ic"),
                "direction": direction,
                "from_llm": bool(item.get("from_llm")),
                "validation_icir": validation_fitness.get("icir"),
                "validation_directional_icir": _finite_or_none(oriented.get("icir")),
                "validation_mean_ic": validation_fitness.get("mean_ic"),
                "validation_coverage": validation_fitness.get("coverage"),
                "passed_filter": verdict.get("passed") if verdicts is not None else None,
                "test_directional_icir": verdict.get("test_directional_icir"),
                "held_on_test": verdict.get("held_on_test"),
            })
        details[method] = candidates
    return details


def _methods(payload: dict[str, Any]) -> list[str]:
    runs = payload.get("paired_runs", [])
    listed = [method for method in methods_of(payload) if any(method in run for run in runs)]
    extra = sorted(
        {
            key for run in runs for key, value in run.items()
            if isinstance(value, dict) and "best_train_abs_icir" in value
        } - set(listed)
    )
    return listed + extra


def _prepare_view_data(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("paired_runs")
    if not isinstance(runs, list):
        raise ValueError(
            "Expected a paired comparison JSON containing a 'paired_runs' array. "
            "Use the comparison_summary.json produced by run_comparison.py."
        )
    methods = _methods(payload)
    prepared_runs = []
    for raw_run in runs:
        if not isinstance(raw_run, dict):
            continue
        run = dict(raw_run)
        for method in methods:
            if isinstance(run.get(method), dict):
                metrics = dict(run[method])
                # Older summaries have no direction-adjusted metric; fall
                # back to the raw signed one so they still render.
                directional = metrics.get("mean_validation_directional_icir_top_k")
                metrics["validation_score_top_k"] = (
                    directional if directional is not None
                    else metrics.get("mean_validation_icir_top_k")
                )
                metrics["validation_score_is_directional"] = directional is not None
                run[method] = metrics
        run["candidate_details"] = _details_for_run(summary_path, run)
        prepared_runs.append(run)
    spare = iter(FALLBACK_COLORS * 3)
    return {
        "dataset": payload.get("dataset", {}),
        "protocol": payload.get("protocol", {}),
        "methods": methods,
        "labels": {method: LABELS.get(method, method) for method in methods},
        "colors": {method: COLORS.get(method) or next(spare) for method in methods},
        "paired_runs": prepared_runs,
    }


# ---------------------------------------------------------------- formatting

def _e(value: Any) -> str:
    return html.escape(str(value))


def _num(value: Any, places: int = 3) -> str:
    value = _finite_or_none(value)
    return "—" if value is None else f"{value:.{places}f}"


def _signed(value: Any, places: int = 3) -> str:
    value = _finite_or_none(value)
    return "—" if value is None else f"{value:+.{places}f}"


def _pct(value: Any, places: int = 0) -> str:
    value = _finite_or_none(value)
    return "—" if value is None else f"{100 * value:.{places}f}%"


def _mean(values: list[Any]) -> float | None:
    values = [value for value in map(_finite_or_none, values) if value is not None]
    return sum(values) / len(values) if values else None


def _td(content: str, classes: list[str] | None = None) -> str:
    """A cell; ``content`` must already be escaped."""

    attribute = f' class="{" ".join(classes)}"' if classes else ""
    return f"<td{attribute}>{content}</td>"


def _html_table(header: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_e(cell.replace(chr(92) + '|', '|'))}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(
            _td(_e(cell), ["neg"] if index >= numeric_from and cell.startswith("-") else None)
            for index, cell in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _dot(view: dict[str, Any], method: str) -> str:
    return f'<i class="dot" style="background:{_e(view["colors"][method])}"></i>'


def _method_runs(view: dict[str, Any], method: str) -> list[tuple[Any, dict[str, Any]]]:
    return [(run.get("seed"), run[method]) for run in view["paired_runs"] if isinstance(run.get(method), dict)]


# ------------------------------------------------------------------ sections

def _kpis(view: dict[str, Any]) -> str:
    runs, methods, labels = view["paired_runs"], view["methods"], view["labels"]
    cards = [
        ("Paired seeds", str(len(runs)), f"{len(methods)} method{'s' if len(methods) != 1 else ''} compared"),
        ("Mean matched budget", _num(_mean([run.get("evaluation_budget") for run in runs]), 0),
         "Unique train evaluations per method and seed"),
    ]
    filtered = []
    for method in methods:
        counts = [metrics["filter"] for _, metrics in _method_runs(view, method) if metrics.get("filter")]
        if counts:
            tested = sum(item.get("num_tests", 0) for item in counts)
            passed = sum(item.get("passed_filter", 0) for item in counts)
            checked = sum(item.get("checked_on_test", 0) for item in counts)
            held = sum(item.get("held_on_test", 0) for item in counts)
            filtered.append((passed, tested, checked, held, method))
    if filtered:
        passed, tested, checked, held, method = max(filtered)
        cards.append((
            "Most formulas passing the filter", f"{passed} / {tested}",
            f"{labels[method]}; {held} of {checked} held on test",
        ))
    validation = [
        (score, method) for method in methods
        if (score := _mean([m.get("validation_score_top_k") for _, m in _method_runs(view, method)])) is not None
    ]
    if validation:
        score, method = max(validation)
        cards.append(("Best mean validation ICIR", _signed(score), f"{labels[method]}; top-k, train direction"))
    llm = [
        (metrics["llm"], method) for method in methods for _, metrics in _method_runs(view, method)
        if metrics.get("llm")
    ]
    if llm:
        calls = sum(item.get("calls") or 0 for item, _ in llm)
        accepted = sum(item.get("accepted") or 0 for item, _ in llm)
        models = sorted({str(item.get("model")) for item, _ in llm if item.get("model")})
        cards.append((
            "LLM proposals accepted", _pct(accepted / calls if calls else None),
            f"{accepted} of {calls} calls; {', '.join(models) or 'model not recorded'}",
        ))
    return "".join(
        f'<article class="card kpi"><div class="kpi-label">{_e(label)}</div>'
        f'<div class="kpi-value">{_e(value)}</div><div class="kpi-note">{_e(note)}</div></article>'
        for label, value, note in cards
    )


def _tables_section(view: dict[str, Any]) -> str:
    text = notes(view)
    header, rows = filter_rows(view)
    table2 = (
        _html_table(header, rows) if rows
        else '<div class="empty">No filter results in this summary (it was run with --no-filter).</div>'
    )
    return (
        f'<div class="section-title"><h2>{_e(FILTER_TITLE)}</h2></div>'
        f'<section class="card"><p class="note">{_e(text["filter"])}</p>{table2}</section>'
        f'<div class="section-title"><h2>{_e(QUALITY_TITLE)}</h2></div>'
        f'<section class="card"><p class="note">{_e(text["quality"])}</p>'
        f'{_html_table(*quality_rows(view))}</section>'
    )


def _legend(view: dict[str, Any]) -> str:
    return '<div class="legend">' + "".join(
        f"<span>{_dot(view, method)}{_e(view['labels'][method])}</span>" for method in view["methods"]
    ) + "</div>"


def _llm_section(view: dict[str, Any]) -> str:
    header = [
        "Method", "Seed", "Model", "Retrieval", "Calls", "Cache hits", "Accepted",
        "After node_type fix", "Rejected", "Rejected by reason", "Acceptance",
        "New formulas evaluated", "Top-k from LLM",
    ]
    rows = []
    for method in view["methods"]:
        seeds = [(seed, metrics) for seed, metrics in _method_runs(view, method) if metrics.get("llm")]
        for seed, metrics in seeds:
            llm = metrics["llm"]
            reasons = ", ".join(f"{key} {value}" for key, value in (llm.get("rejected_by_reason") or {}).items())
            rows.append([
                view["labels"][method], str(seed), str(llm.get("model") or "—"),
                "yes" if llm.get("retrieval", True) else "no",
                str(llm.get("calls", "—")), str(llm.get("cache_hits", "—")), str(llm.get("accepted", "—")),
                str(llm.get("repaired_node_types", "—")), str(llm.get("rejected", "—")), reasons or "none",
                _pct(llm.get("acceptance_rate")), str(llm.get("unique_llm_evaluations", "—")),
                str(metrics.get("top_k_from_llm", "—")),
            ])
        if len(seeds) > 1:
            total = lambda key: sum(int(m["llm"].get(key) or 0) for _, m in seeds)  # noqa: E731
            calls, accepted = total("calls"), total("accepted")
            rows.append([
                view["labels"][method], "all", "", "", str(calls), str(total("cache_hits")),
                str(accepted), str(total("repaired_node_types")), str(total("rejected")), "",
                _pct(accepted / calls if calls else None), str(total("unique_llm_evaluations")),
                str(sum(int(m.get("top_k_from_llm") or 0) for _, m in seeds)),
            ])
    if not rows:
        return ""
    return (
        '<div class="section-title"><h2>LLM operator</h2>'
        '<span class="hint">Each call asks the model for one variation of a parent formula; '
        "rejected answers fall back to a random mutation.</span></div>"
        f'<section class="card">{_html_table(header, rows, numeric_from=4)}</section>'
    )


def _per_seed_section(view: dict[str, Any]) -> str:
    header = [
        "Seed", "Method", "Budget", "Best train |ICIR|", "Mean top-k validation ICIR",
        "Direction agreement", "Top-k diversity", "Passed filter", "Held on test", "Runtime",
    ]
    body = []
    for run in view["paired_runs"]:
        present = [method for method in view["methods"] if isinstance(run.get(method), dict)]
        scores = {
            method: _finite_or_none(run[method].get("validation_score_top_k")) for method in present
        }
        best = max((value for value in scores.values() if value is not None), default=None)
        for index, method in enumerate(present):
            metrics = run[method]
            counts = metrics.get("filter") or {}
            winner = best is not None and scores[method] == best and len(present) > 1
            cells = [
                (str(run.get("seed")) if index == 0 else "", ""),
                (f"{_dot(view, method)}{_e(view['labels'][method])}", "html"),
                (str(run.get("evaluation_budget", "—")) if index == 0 else "", ""),
                (_num(metrics.get("best_train_abs_icir")), ""),
                (_signed(metrics.get("validation_score_top_k")), "win" if winner else ""),
                (_pct(metrics.get("train_validation_direction_agreement_rate")), ""),
                (_num((metrics.get("top_k_expression_diversity") or {}).get(
                    "mean_pairwise_operator_jaccard_distance")), ""),
                (str(counts.get("passed_filter", "—")), ""),
                (str(counts.get("held_on_test", "—")), ""),
                (f"{_num(metrics.get('elapsed_seconds'), 1)} s", ""),
            ]
            tds = []
            for text, kind in cells:
                classes = []
                if kind == "win":
                    classes.append("winner")
                if text.startswith("-"):
                    classes.append("neg")
                tds.append(_td(text if kind == "html" else _e(text), classes))
            row_class = ' class="group-start"' if index == 0 else ""
            body.append(f'<tr{row_class}>{"".join(tds)}</tr>')
    if not body:
        return '<section class="card empty">No paired runs were found in this JSON.</section>'
    head = "".join(f"<th>{_e(cell)}</th>" for cell in header)
    return (
        '<div class="section-title"><h2>Per-seed metrics</h2><span class="hint">'
        "Bold green marks the best mean validation ICIR of each seed.</span></div>"
        f'<section class="card"><div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div></section>'
    )


def _formula_sections(view: dict[str, Any]) -> str:
    blocks = []
    for run in view["paired_runs"]:
        for method in view["methods"]:
            formulas = (run.get("candidate_details") or {}).get(method) or []
            if not formulas:
                continue
            passed = sum(1 for item in formulas if item.get("passed_filter"))
            filtered = any(item.get("passed_filter") is not None for item in formulas)
            summary = f"Seed {run.get('seed')} · {view['labels'][method]} · top {len(formulas)}"
            if filtered:
                summary += f" · {passed} passed the filter"
            items = []
            for rank, item in enumerate(formulas, start=1):
                tags = []
                if item.get("from_llm"):
                    tags.append('<span class="tag llm">LLM</span>')
                if item.get("passed_filter"):
                    held = item.get("held_on_test")
                    tags.append('<span class="tag pass">passed filter</span>')
                    tags.append(
                        f'<span class="tag {"pass" if held else "fail"}">test ICIR '
                        f'{_e(_signed(item.get("test_directional_icir")))}</span>'
                    )
                elif item.get("passed_filter") is False:
                    tags.append('<span class="tag">did not pass</span>')
                validation = item.get("validation_directional_icir")
                if validation is None:
                    validation = item.get("validation_icir")
                items.append(
                    f'<div class="formula"><div class="formula-top"><span>#{rank} · train |ICIR| '
                    f'{_e(_num(item.get("train_score")))} · validation ICIR (train direction) '
                    f'{_e(_signed(validation))}</span><span>{"".join(tags)}</span></div>'
                    f'<code>{_e(item.get("expression", ""))}</code></div>'
                )
            blocks.append(
                f'<details class="card"><summary>{_dot(view, method)}{_e(summary)}</summary>'
                f'<div class="formula-list">{"".join(items)}</div></details>'
            )
    if not blocks:
        return (
            '<div class="card empty">No per-run result files were found beside the summary. '
            "Metrics, tables and charts are still available.</div>"
        )
    return "".join(blocks)


def _protocol_note(view: dict[str, Any]) -> str:
    protocol, dataset = view["protocol"], view["dataset"]
    config = protocol.get("evolution_config") or {}
    statistical = protocol.get("statistical_filter") or {}
    parts = [
        f"Objective {protocol.get('fitness_objective', '—')} on the {protocol.get('train_split', 'train')} split; "
        f"validation used for selection: {'yes' if protocol.get('validation_used_for_selection') else 'no'}.",
        f"Horizon {config.get('horizon', '—')} days, maximum depth {config.get('max_depth', '—')}, "
        f"population {config.get('population_size', '—')}, {config.get('generations', '—')} generations, "
        f"seed pool {protocol.get('evolution_seed_pool', '—')}.",
        f"Evaluation budget: {protocol.get('evaluation_budget', '—')}.",
    ]
    if statistical:
        parts.append(
            f"Statistical filter: {statistical.get('method', '—')} test, Benjamini–Hochberg q = "
            f"{statistical.get('q', '—')}; test split checked once: "
            f"{'yes' if statistical.get('test_split_checked') else 'no'}."
        )
    llm = protocol.get("evolution_llm_config") or {}
    if llm:
        parts.append(
            f"LLM: {llm.get('llm_backend', '—')} / {llm.get('llm_model', '—')}, "
            f"share of children {llm.get('llm_rate', '—')}; ablation seeds without retrieval: "
            f"{', '.join(map(str, protocol.get('ablation_seeds') or [])) or 'none'}."
        )
    parts.append(f"Dataset SHA-256 {dataset.get('sha256') or 'not recorded'}.")
    return " ".join(parts)


# --------------------------------------------------------------------- page

STYLE = """
:root{--ink:#162b35;--muted:#60757e;--paper:#f2f5f2;--card:#fff;--line:#dce5e2;--green:#087e72;--red:#b94e51;--shadow:0 12px 34px rgba(30,59,66,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1440px;margin:0 auto;padding:30px 32px 64px}header{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:24px}
.eyebrow{color:var(--green);font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(28px,4vw,44px);letter-spacing:-.045em;line-height:1.1;margin:8px 0 12px}.sub{color:var(--muted);max-width:850px;margin:0}
.badge{background:var(--ink);color:#f4fbf8;padding:10px 14px;border-radius:999px;font-size:12px;font-weight:700;white-space:nowrap}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 26px}.chip{border:1px solid var(--line);background:#ffffffaa;border-radius:999px;padding:6px 11px;color:#46606a;font-size:12px;max-width:100%;overflow-wrap:anywhere}.chip strong{color:var(--ink)}
.section-title{display:flex;justify-content:space-between;align-items:end;margin:30px 2px 12px;gap:12px;flex-wrap:wrap}h2{font-size:18px;letter-spacing:-.02em;margin:0}.hint{font-size:12px;color:var(--muted)}
.grid{display:grid;gap:14px}.kpis{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.kpi{padding:18px 20px;min-height:118px}.kpi-label{color:var(--muted);font-size:12px;font-weight:700}.kpi-value{font-size:30px;letter-spacing:-.04em;font-weight:750;margin:9px 0 2px}.kpi-note{font-size:11px;color:var(--muted)}
.note{margin:0;padding:16px 18px 4px;color:var(--muted);font-size:12px}
.charts{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}.chart-card{padding:18px 18px 12px;min-width:0}.chart-title{font-weight:750;font-size:14px;margin-bottom:3px}.chart-note{font-size:11px;color:var(--muted);min-height:32px}canvas{width:100%;height:250px;display:block}
.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:#f8faf9}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}tbody tr:last-child td{border-bottom:0}tr.group-start td{border-top:2px solid var(--line)}.winner{color:var(--green);font-weight:800}.neg{color:var(--red)}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:8px;vertical-align:middle}
details{padding:15px 18px}summary{cursor:pointer;color:var(--ink);font-size:13px;font-weight:750}.formula-list{display:grid;gap:9px;margin-top:12px}.formula{padding:12px;background:#f7faf8;border:1px solid var(--line);border-radius:11px}.formula-top{display:flex;justify-content:space-between;gap:10px;margin-bottom:6px;font-size:11px;color:var(--muted);flex-wrap:wrap}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere;color:#173c46}
.tag{display:inline-block;margin-left:6px;padding:1px 8px;border-radius:999px;border:1px solid var(--line);font-size:11px;color:var(--muted)}.tag.llm{border-color:#4777a7;color:#4777a7}.tag.pass{border-color:var(--green);color:var(--green)}.tag.fail{border-color:var(--red);color:var(--red)}
.footnote{padding:16px 18px;color:var(--muted);font-size:12px;margin-top:18px}.empty{color:var(--muted);padding:20px}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:5px}
@media(max-width:680px){.wrap{padding:22px 16px 40px}header{display:block}.badge{display:inline-block;margin-top:16px}.kpi-value{font-size:25px}}
@media print{body{background:#fff}.wrap{max-width:none;padding:0}.card{box-shadow:none;break-inside:avoid}}
"""

SCRIPT = """
const D=JSON.parse(document.getElementById('report-data').textContent);
const runs=D.paired_runs||[],M=D.methods||[],COL=D.colors||{},GRID='#e4ece9',MUTED='#71848a';
const val=(r,m,k)=>{const v=r[m]?.[k];return v===null||v===undefined||!Number.isFinite(Number(v))?null:Number(v)};
const mean=a=>{a=a.filter(x=>x!==null);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null};
function setup(canvas){const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(300,rect.width*ratio);canvas.height=250*ratio;const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);return{ctx,w:canvas.width/ratio,h:250}}
function axis(ctx,w,pad,lo,hi,y,fmt){ctx.font='11px system-ui';ctx.strokeStyle=GRID;ctx.fillStyle=MUTED;ctx.lineWidth=1;for(let i=0;i<5;i++){const v=hi-i*(hi-lo)/4,yy=y(v);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillText(fmt(v,hi-lo),2,yy+4)}}
function range(values){let lo=Math.min(0,...values),hi=Math.max(0,...values);if(hi-lo<.02){hi+=.01;lo-=.01}return[lo,hi]}
function drawLine(canvas,labels,series){const{ctx,w,h}=setup(canvas),pad={l:44,r:14,t:16,b:36},all=series.flatMap(s=>s.values.filter(v=>v!==null));if(!all.length)return;const[lo,hi]=range(all);const x=i=>pad.l+(labels.length<2?(w-pad.l-pad.r)/2:i*(w-pad.l-pad.r)/(labels.length-1)),y=v=>pad.t+(hi-v)*(h-pad.t-pad.b)/(hi-lo);axis(ctx,w,pad,lo,hi,y,v=>v.toFixed(2));labels.forEach((label,i)=>ctx.fillText(label,x(i)-8,h-10));for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2.5;ctx.beginPath();let prev=false;s.values.forEach((v,i)=>{if(v===null){prev=false;return}if(prev)ctx.lineTo(x(i),y(v));else ctx.moveTo(x(i),y(v));prev=true});ctx.stroke();s.values.forEach((v,i)=>{if(v===null)return;ctx.beginPath();ctx.fillStyle=s.color;ctx.arc(x(i),y(v),4,0,Math.PI*2);ctx.fill()})}}
function drawBars(canvas,groups,fmt){const{ctx,w,h}=setup(canvas),pad={l:46,r:12,t:16,b:40},all=groups.flatMap(g=>M.map(m=>g.values[m]).filter(v=>v!==null&&v!==undefined));if(!all.length)return;const[lo,hi]=range(all);const y=v=>pad.t+(hi-v)*(h-pad.t-pad.b)/(hi-lo),base=y(0),groupW=(w-pad.l-pad.r)/Math.max(groups.length,1),barW=Math.max(4,Math.min(24,groupW*.8/Math.max(M.length,1)-3));axis(ctx,w,pad,lo,hi,y,fmt);groups.forEach((g,i)=>{const cx=pad.l+groupW*(i+.5),start=cx-(M.length*(barW+3))/2;M.forEach((m,j)=>{const v=g.values[m];if(v===null||v===undefined)return;const yy=y(v);ctx.fillStyle=COL[m];ctx.fillRect(start+j*(barW+3),Math.min(yy,base),barW,Math.max(1,Math.abs(base-yy)))});ctx.fillStyle=MUTED;ctx.fillText(g.label,cx-ctx.measureText(g.label).width/2,h-12)})}
function draw(){
 drawLine(document.getElementById('validation-chart'),runs.map(r=>String(r.seed)),M.map(m=>({color:COL[m],values:runs.map(r=>val(r,m,'validation_score_top_k'))})));
 const avg=k=>Object.fromEntries(M.map(m=>[m,mean(runs.map(r=>val(r,m,k)))]));
 drawBars(document.getElementById('performance-chart'),[{label:'Train |ICIR|',values:avg('mean_train_abs_icir_top_k')},{label:'Validation ICIR',values:avg('validation_score_top_k')}],v=>v.toFixed(2));
 drawBars(document.getElementById('runtime-chart'),runs.map(r=>({label:'Seed '+r.seed,values:Object.fromEntries(M.map(m=>[m,val(r,m,'elapsed_seconds')]))})),(v,span)=>v.toFixed(span<5?1:0)+'s');
}
draw();window.addEventListener('resize',draw);
"""


def _embedded_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")


def render_html(view_data: dict[str, Any], title: str | None = None) -> str:
    dataset = view_data.get("dataset", {})
    protocol = view_data.get("protocol", {})
    view_data = {
        "methods": [], "labels": {}, "colors": {}, **view_data,
    }
    labels = [view_data["labels"][method] for method in view_data["methods"]]
    statistical = protocol.get("statistical_filter") or {}
    badge = (
        "TEST SPLIT CHECKED ONCE" if statistical.get("test_split_checked")
        else "VALIDATION ONLY"
    )
    llm = protocol.get("evolution_llm_config") or {}
    chips = [
        ("Dataset", dataset.get("path", "Research dataset")),
        ("Dates", f"{dataset.get('date_min', '—')} → {dataset.get('date_max', '—')}"),
        ("Horizon", f"{(protocol.get('evolution_config') or {}).get('horizon', '—')}d"),
        ("Top K", protocol.get("top_k", "—")),
    ]
    if llm:
        chips.append(("LLM", f"{llm.get('llm_model') or llm.get('llm_backend', '—')}"))
    chips.append(("Fingerprint", str(dataset.get("sha256", ""))[:12] or "not recorded"))
    chip_html = "".join(f'<span class="chip"><strong>{_e(k)}</strong> {_e(v)}</span>' for k, v in chips)
    compared = ", ".join(labels[:-1]) + (" and " if len(labels) > 1 else "") + (labels[-1] if labels else "")
    page_title = title or "Aldien · Search comparison"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{_e(page_title)}</title>
<style>{STYLE}</style>
</head>
<body><main class="wrap">
<header><div><div class="eyebrow">Aldien · Alpha Discovery Engine</div><h1>Search experiment report</h1>
<p class="sub">Compares {_e(compared or "the search methods")} on the same market data, train objective and per-seed evaluation budget. Formulas are selected on the train split only; validation is reported, and formulas that pass the statistical filter are checked once on the test split.</p></div>
<div class="badge">{_e(badge)}</div></header>
<div class="meta">{chip_html}</div>
<div class="section-title"><h2>At a glance</h2><span class="hint">{len(view_data["paired_runs"])} paired seeds · {_e(dataset.get("rows", "—"))} dataset rows</span></div>
<section class="grid kpis">{_kpis(view_data)}</section>
{_tables_section(view_data)}
<div class="section-title"><h2>Across-seed comparison</h2><span class="hint">Methods run on only some seeds (the ablation) appear as separate points.</span></div>
<section class="grid charts">
 <article class="card chart-card"><div class="chart-title">Validation ICIR by seed</div><div class="chart-note">Mean validation ICIR of each method's top-K train-ranked formulas, oriented by each formula's train direction.</div><canvas id="validation-chart"></canvas>{_legend(view_data)}</article>
 <article class="card chart-card"><div class="chart-title">Mean method performance</div><div class="chart-note">Average top-K train |ICIR| and direction-adjusted validation ICIR across seeds.</div><canvas id="performance-chart"></canvas>{_legend(view_data)}</article>
 <article class="card chart-card"><div class="chart-title">Runtime per seed</div><div class="chart-note">Elapsed seconds for each search run, including finalist validation.</div><canvas id="runtime-chart"></canvas>{_legend(view_data)}</article>
</section>
{_llm_section(view_data)}
{_per_seed_section(view_data)}
<div class="section-title"><h2>Top formulas by seed</h2><span class="hint">From the per-run result files; filter and test results from the matching _filtered.json files.</span></div>
<section class="grid">{_formula_sections(view_data)}</section>
<section class="card footnote">{_e(_protocol_note(view_data))}</section>
</main>
<script id="report-data" type="application/json">{_embedded_json(view_data)}</script>
<script>{SCRIPT}</script>
</body></html>
"""


def generate_report(input_path: Path, output_path: Path, title: str | None = None) -> Path:
    payload = _load_json(input_path)
    view_data = _prepare_view_data(input_path, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(view_data, title=title), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
