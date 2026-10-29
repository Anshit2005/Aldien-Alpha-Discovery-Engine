"""Build a self-contained presentation report from a paired search summary."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


EVOLUTION = "evolutionary"
RANDOM = "random"
METHODS = (EVOLUTION, RANDOM)
COLORS = {EVOLUTION: "#087f78", RANDOM: "#e18a48"}
LABELS = {EVOLUTION: "Evolutionary", RANDOM: "Random"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def fmt(value: Any, places: int = 3) -> str:
    if not finite(value):
        return "N/A"
    return f"{value:+.{places}f}" if value < 0 else f"{value:.{places}f}"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def root_operator(expression: str) -> str:
    return expression.split("(", 1)[0] if "(" in expression else "TERMINAL"


def chart_grouped(
    labels: list[str],
    series: list[tuple[str, list[float | None], str]],
    *,
    min_zero: bool = True,
    suffix: str = "",
    decimals: int = 2,
) -> str:
    """Create responsive inline SVG grouped bars with native hover titles."""
    width, height = 920, 340
    left, right, top, bottom = 66, 20, 26, 58
    values = [v for _, vs, _ in series for v in vs if finite(v)]
    if not values:
        return '<p class="empty">No finite values available for this chart.</p>'
    low = min([0.0, *values]) if min_zero else min(values)
    high = max([0.0, *values])
    span = max(high - low, 0.02)
    low -= span * 0.08
    high += span * 0.12
    plot_h, plot_w = height - top - bottom, width - left - right

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_h

    zero = y(0.0)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for i in range(5):
        val = high - i * (high - low) / 4
        yy = y(val)
        parts.append(f'<line class="gridline" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="axis" x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{val:.{decimals}f}{suffix}</text>')
    group_w = plot_w / max(1, len(labels))
    n_series = len(series)
    bar_w = min(38, group_w * 0.72 / n_series)
    for index, label in enumerate(labels):
        center = left + group_w * (index + 0.5)
        parts.append(f'<text class="axis x-label" x="{center:.1f}" y="{height-22}" text-anchor="middle">{esc(label)}</text>')
        for series_index, (name, vals, color) in enumerate(series):
            value = vals[index] if index < len(vals) else None
            if not finite(value):
                parts.append(f'<text class="na-label" x="{center:.1f}" y="{height-39}" text-anchor="middle">N/A</text>')
                continue
            x = center + (series_index - (n_series - 1) / 2) * (bar_w + 5) - bar_w / 2
            yy = y(float(value))
            rect_y, rect_h = min(yy, zero), max(1.5, abs(zero - yy))
            title = esc(f"{label} · {name}: {float(value):.{decimals}f}{suffix}")
            parts.append(f'<rect class="bar" x="{x:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" rx="4" fill="{color}"><title>{title}</title></rect>')
            label_y = rect_y - 5 if float(value) >= 0 else rect_y + rect_h + 14
            parts.append(f'<text class="bar-value" x="{x+bar_w/2:.1f}" y="{label_y:.1f}" text-anchor="middle">{float(value):.{decimals}f}</text>')
    parts.append(f'<line class="zero-line" x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}"/>')
    parts.append('</svg>')
    return "".join(parts)


def chart_generation(histories: list[list[dict[str, Any]]]) -> str:
    width, height = 920, 350
    left, right, top, bottom = 68, 25, 25, 54
    generations = sorted({int(row["generation"]) for history in histories for row in history})
    stats = []
    for generation in generations:
        values = [float(row["best_train_score"]) for history in histories for row in history
                  if int(row["generation"]) == generation and finite(row.get("best_train_score"))]
        stats.append((generation, mean(values), sample_sd(values) or 0.0))
    if not stats:
        return '<p class="empty">Generation history is unavailable.</p>'
    high = max(m + s for _, m, s in stats if m is not None) * 1.12
    low = 0.0
    pw, ph = width-left-right, height-top-bottom
    x = lambda i: left + i * pw / max(1, len(stats)-1)
    y = lambda v: top + (high-v) * ph / max(high-low, 1e-9)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for i in range(5):
        v = high - i * high / 4
        yy = y(v)
        out.append(f'<line class="gridline" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        out.append(f'<text class="axis" x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{v:.2f}</text>')
    upper, lower, line = [], [], []
    for i, (gen, avg, sd) in enumerate(stats):
        assert avg is not None
        upper.append(f'{x(i):.1f},{y(avg+sd):.1f}')
        lower.append(f'{x(i):.1f},{y(max(0,avg-sd)):.1f}')
        line.append(f'{x(i):.1f},{y(avg):.1f}')
    out.append(f'<polygon class="sd-band" points="{" ".join(upper + list(reversed(lower)))}"/>')
    out.append(f'<polyline class="gen-line" points="{" ".join(line)}"/>')
    for i, (gen, avg, sd) in enumerate(stats):
        assert avg is not None
        out.append(f'<line class="errorbar" x1="{x(i):.1f}" y1="{y(avg+sd):.1f}" x2="{x(i):.1f}" y2="{y(max(0,avg-sd)):.1f}"/>')
        out.append(f'<circle class="gen-point" cx="{x(i):.1f}" cy="{y(avg):.1f}" r="5"><title>Generation {gen}: mean {avg:.4f}, SD {sd:.4f}, n={len(histories)}</title></circle>')
        out.append(f'<text class="axis x-label" x="{x(i):.1f}" y="{height-20}" text-anchor="middle">{gen}</text>')
    out.append(f'<text class="axis" x="{width/2}" y="{height-3}" text-anchor="middle">Generation (0 = initial population)</text>')
    out.append('</svg>')
    return "".join(out)


def legend(series: list[tuple[str, list[float | None], str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{color}"></i>{esc(name)}</span>'
        for name, _, color in series
    ) + '</div>'


def td(value: Any, *, places: int = 3, cls: str = "") -> str:
    text = fmt(value, places) if isinstance(value, (int, float)) or value is None else esc(value)
    return f'<td class="{cls}">{text}</td>'


def build(summary_path: Path, output_path: Path) -> None:
    summary = load(summary_path)
    paired = summary.get("paired_runs", [])
    if not paired:
        raise ValueError("Comparison summary has no paired_runs")
    data_root = summary_path.parent
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    histories = []
    for pair in paired:
        seed = int(pair["seed"])
        for method in METHODS:
            filename = pair.get("result_files", {}).get(method)
            if filename:
                path = Path(filename)
                if not path.is_file():
                    path = data_root / path.name
                if path.is_file():
                    result = load(path)
                    runs[(seed, method)] = result
                    if method == EVOLUTION:
                        histories.append(result.get("generations", []))

    def m(seed: int, method: str) -> dict[str, Any]:
        return next(row[method] for row in paired if int(row["seed"]) == seed)

    def result(seed: int, method: str) -> dict[str, Any]:
        return runs.get((seed, method), {})

    def candidates(seed: int, method: str) -> list[dict[str, Any]]:
        return result(seed, method).get("best", [])

    def validation_by_expr(seed: int, method: str) -> dict[str, dict[str, Any]]:
        return {row.get("expression"): row for row in result(seed, method).get("validation", [])}

    def valid_validation(seed: int, method: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        vmap = validation_by_expr(seed, method)
        rows = []
        for cand in candidates(seed, method):
            val = vmap.get(cand.get("expression"), {}).get("fitness", {})
            if finite(val.get("icir")):
                rows.append((cand, val))
        return rows

    def valid_direction_agreement(seed: int, method: str) -> float | None:
        rows = valid_validation(seed, method)
        if not rows:
            return None
        return sum((float(c["fitness"]["icir"]) > 0) == (float(v["icir"]) > 0)
                   for c, v in rows) / len(rows)

    seeds = [int(row["seed"]) for row in paired]
    valid_seeds_by_method = {
        method: [s for s in seeds if valid_validation(s, method)] for method in METHODS
    }
    paired_valid_seeds = [s for s in seeds if all(valid_validation(s, method) for method in METHODS)]

    # Table 1 and its training charts.
    t1_rows = []
    for seed in seeds:
        evo, rnd = m(seed, EVOLUTION), m(seed, RANDOM)
        def win_class(a: Any, b: Any) -> tuple[str, str]:
            if not finite(a) or not finite(b):
                return "", ""
            return ("winner", "") if a > b else (("", "winner") if b > a else ("", ""))
        b1, b2 = win_class(evo.get("best_train_abs_icir"), rnd.get("best_train_abs_icir"))
        q1, q2 = win_class(evo.get("mean_train_abs_icir_top_k"), rnd.get("mean_train_abs_icir_top_k"))
        t1_rows.append(f'<tr><th scope="row">{seed}</th>{td(evo.get("best_train_abs_icir"),cls=b1)}{td(rnd.get("best_train_abs_icir"),cls=b2)}{td(evo.get("mean_train_abs_icir_top_k"),cls=q1)}{td(rnd.get("mean_train_abs_icir_top_k"),cls=q2)}</tr>')

    best_series = [(LABELS[method], [m(s, method).get("best_train_abs_icir") for s in seeds], COLORS[method]) for method in METHODS]
    mean_series = [(LABELS[method], [m(s, method).get("mean_train_abs_icir_top_k") for s in seeds], COLORS[method]) for method in METHODS]

    # Generation-wise means and sample standard deviations.
    generation_rows = []
    gen_ids = sorted({int(row["generation"]) for hist in histories for row in hist})
    gen_means, gen_sds = [], []
    gen0 = None
    for g in gen_ids:
        vals = [float(row["best_train_score"]) for hist in histories for row in hist
                if int(row["generation"]) == g and finite(row.get("best_train_score"))]
        avg, sd = mean(vals), sample_sd(vals)
        if g == 0:
            gen0 = avg
        improvement = avg - gen0 if avg is not None and gen0 is not None else None
        gen_means.append(avg)
        gen_sds.append(sd)
        generation_rows.append(f'<tr><th scope="row">{g}</th>{td(avg)}{td(sd)}{td(improvement)}{td(100*improvement/gen0 if improvement is not None and gen0 else None,places=1)}<td>{len(vals)}</td></tr>')

    def sign_count(signs: list[str]) -> str:
        if not signs:
            return "N/A"
        counts = Counter(signs)
        return f"+ {counts.get('+',0)} · − {counts.get('−',0)} · 0 {counts.get('0',0)} of {len(signs)}"

    # Training/validation table, per-seed sign table, diversity, and cost tables.
    t2_rows, sign_rows, diversity_rows, cost_rows = [], [], [], []
    sign_data = {method: [] for method in METHODS}
    diversity_data = {method: [] for method in METHODS}
    runtime_data = {method: [] for method in METHODS}
    for seed in seeds:
        for method in METHODS:
            met = m(seed, method)
            valid = valid_validation(seed, method)
            agreement = valid_direction_agreement(seed, method)
            # Show per-top-10 mean values from the recorded summary. Missing stays N/A.
            t2_rows.append(
                f'<tr><th scope="row">{seed}</th><td>{LABELS[method]}</td>'
                f'{td(met.get("mean_train_abs_icir_top_k"))}'
                f'{td(met.get("best_validation_icir_top_k"))}'
                f'{td(met.get("mean_validation_icir_top_k"))}'
                f'{td(met.get("best_validation_directional_icir_top_k"),cls="directional")}'
                f'{td(met.get("mean_validation_directional_icir_top_k"),cls="directional")}'
                f'{td(met.get("train_to_validation_abs_icir_degradation"))}</tr>'
            )
            training_signs = ["+" if c["fitness"]["icir"] > 0 else "−" if c["fitness"]["icir"] < 0 else "0"
                              for c in candidates(seed, method) if finite(c.get("fitness", {}).get("icir"))]
            train_desc = sign_count(training_signs)
            val_signs = ["+" if v["icir"] > 0 else "−" if v["icir"] < 0 else "0" for _, v in valid]
            val_desc = sign_count(val_signs) if val_signs else "N/A · no valid IC"
            agreement_text = "N/A" if agreement is None else f"{agreement * 100:.0f}%"
            sign_rows.append(
                f'<tr><th scope="row">{seed}</th><td>{LABELS[method]}</td>'
                f'<td>{esc(train_desc)}</td><td>{esc(val_desc)}</td>'
                f'<td>{agreement_text}</td></tr>'
            )
            sign_data[method].append(agreement)
            roots = Counter(root_operator(c["expression"]) for c in candidates(seed, method))
            root_text = ", ".join(f'{key} ({value})' for key, value in sorted(roots.items())) or "N/A"
            diversity = met.get("top_k_expression_diversity", {})
            diversity_rows.append(
                f'<tr><th scope="row">{seed}</th><td>{LABELS[method]}</td><td>{esc(root_text)}</td>'
                f'{td(diversity.get("mean_pairwise_operator_jaccard_distance"))}'
                f'<td>{diversity.get("unique_expression_count", "N/A")}/10</td></tr>'
            )
            diversity_data[method].append(diversity.get("mean_pairwise_operator_jaccard_distance"))
            runtime = met.get("elapsed_seconds")
            runtime_data[method].append(runtime)
            best_val = met.get("best_validation_icir_top_k")
            cost_rows.append(
                f'<tr><th scope="row">{seed}</th><td>{LABELS[method]}</td>'
                f'<td>{met.get("unique_evaluations", "N/A")}</td>{td(runtime,places=1)}'
                f'{td(met.get("best_train_abs_icir"))}{td(best_val)}</tr>'
            )

    # Table 3 chart uses recomputed agreement, so invalid validation is truly N/A.
    direction_series = [(LABELS[method], sign_data[method], COLORS[method]) for method in METHODS]
    diversity_series = [(LABELS[method], diversity_data[method], COLORS[method]) for method in METHODS]
    runtime_series = [(LABELS[method], runtime_data[method], COLORS[method]) for method in METHODS]

    root_categories = sorted({root_operator(c["expression"])
                              for method in METHODS for seed in seeds for c in candidates(seed, method)})
    root_series = []
    for method in METHODS:
        totals = Counter(root_operator(c["expression"]) for seed in seeds for c in candidates(seed, method))
        # Average number of top-10 candidates per seed whose root is each operator.
        root_series.append({"name": LABELS[method], "color": COLORS[method],
                            "values": [totals.get(root, 0) / len(seeds) for root in root_categories]})

    # Aggregate rows: method averages use all valid values; validation columns disclose sample sizes.
    aggregate_metrics = {
        "Mean best train |ICIR|": ("best_train_abs_icir", False),
        "Mean top-10 train |ICIR|": ("mean_train_abs_icir_top_k", False),
        "Mean best raw validation ICIR": ("best_validation_icir_top_k", True),
        "Mean top-10 raw validation ICIR": ("mean_validation_icir_top_k", True),
        "Mean best direction-adjusted validation ICIR": ("best_validation_directional_icir_top_k", True),
        "Mean top-10 direction-adjusted validation ICIR": ("mean_validation_directional_icir_top_k", True),
        "Direction agreement rate": ("__agreement__", True),
        "Mean operator Jaccard distance": ("__diversity__", False),
        "Mean unique evaluations": ("unique_evaluations", False),
        "Mean runtime (seconds)": ("elapsed_seconds", False),
        "Train−validation |ICIR| difference": ("train_to_validation_abs_icir_degradation", True),
    }
    aggregate_rows = []
    aggregate_values: dict[str, dict[str, float | None]] = {method: {} for method in METHODS}
    for label, (key, validation_metric) in aggregate_metrics.items():
        values_by_method = {}
        counts_by_method = {}
        for method in METHODS:
            vals = []
            for seed in seeds:
                if key == "__agreement__":
                    value = valid_direction_agreement(seed, method)
                elif key == "__diversity__":
                    value = m(seed, method).get("top_k_expression_diversity", {}).get("mean_pairwise_operator_jaccard_distance")
                else:
                    value = m(seed, method).get(key)
                if finite(value):
                    vals.append(float(value))
            values_by_method[method] = mean(vals)
            counts_by_method[method] = len(vals)
            aggregate_values[method][key] = mean(vals)
        n_e, n_r = counts_by_method[EVOLUTION], counts_by_method[RANDOM]
        a, b = values_by_method[EVOLUTION], values_by_method[RANDOM]
        if not finite(a) or not finite(b):
            verdict = "Insufficient valid values"
        elif key in ("mean_validation_directional_icir_top_k", "__agreement__"):
            verdict = "Random higher" if b > a else "Evolution higher" if a > b else "Tie"
        elif key in ("__diversity__",):
            verdict = "Random more diverse" if b > a else "Evolution more diverse" if a > b else "Tie"
        elif key == "elapsed_seconds":
            verdict = "Evolution faster" if a < b else "Random faster" if b < a else "Tie"
        elif key == "unique_evaluations":
            verdict = "Matched budget"
        else:
            verdict = "Evolution higher" if a > b else "Random higher" if b > a else "Tie"
        if key == "__agreement__":
            cell_a = "N/A" if not finite(a) else f"{a * 100:.1f}%"
            cell_b = "N/A" if not finite(b) else f"{b * 100:.1f}%"
        else:
            cell_a = fmt(a)
            cell_b = fmt(b)
        aggregate_rows.append(
            f'<tr><th scope="row">{esc(label)}</th><td>{cell_a}</td><td class="n-cell">n={n_e}</td>'
            f'<td>{cell_b}</td><td class="n-cell">n={n_r}</td><td>{esc(verdict)}</td></tr>'
        )

    # A paired validation comparison over identical seeds.
    paired_direction = {
        method: mean([m(seed, method).get("mean_validation_directional_icir_top_k")
                      for seed in paired_valid_seeds
                      if finite(m(seed, method).get("mean_validation_directional_icir_top_k"))])
        for method in METHODS
    }

    def win_count(key: str) -> tuple[int, int]:
        evowins = randomwins = 0
        for seed in seeds:
            a, b = m(seed, EVOLUTION).get(key), m(seed, RANDOM).get(key)
            if finite(a) and finite(b):
                evowins += a > b
                randomwins += b > a
        return evowins, randomwins

    top_wins = win_count("mean_train_abs_icir_top_k")
    best_wins = win_count("best_train_abs_icir")
    evo_train_top = mean([m(s, EVOLUTION)["mean_train_abs_icir_top_k"] for s in seeds])
    rnd_train_top = mean([m(s, RANDOM)["mean_train_abs_icir_top_k"] for s in seeds])
    evo_div = mean([m(s, EVOLUTION)["top_k_expression_diversity"]["mean_pairwise_operator_jaccard_distance"] for s in seeds])
    rnd_div = mean([m(s, RANDOM)["top_k_expression_diversity"]["mean_pairwise_operator_jaccard_distance"] for s in seeds])
    evo_valid_mean = mean([m(s, EVOLUTION).get("mean_validation_directional_icir_top_k") for s in valid_seeds_by_method[EVOLUTION] if finite(m(s, EVOLUTION).get("mean_validation_directional_icir_top_k"))])
    rnd_valid_mean = mean([m(s, RANDOM).get("mean_validation_directional_icir_top_k") for s in valid_seeds_by_method[RANDOM] if finite(m(s, RANDOM).get("mean_validation_directional_icir_top_k"))])

    dataset = summary.get("dataset", {})
    protocol = summary.get("protocol", {})
    cfg = protocol.get("evolution_config", {})
    first_evo = result(seeds[0], EVOLUTION).get("config", {})
    methodology = [
        ("Search space", "Symbolic alpha DSL"), ("Training split", protocol.get("train_split", "train")),
        ("Validation split", protocol.get("validation_split", "validation")),
        ("Objective", "Absolute training ICIR"), ("Finalists", f"Top {protocol.get('top_k', 10)} per run"),
        ("Seeds", ", ".join(map(str, seeds))), ("Evaluation budget", "Matched unique train evaluations per seed"),
        ("Population", cfg.get("population_size", first_evo.get("population_size", "N/A"))),
        ("Generations", cfg.get("generations", first_evo.get("generations", "N/A"))),
        ("Elite count", cfg.get("elite_count", first_evo.get("elite_count", "N/A"))),
        ("Tournament size", first_evo.get("tournament_size", "N/A")),
        ("Mutation / crossover", f"{cfg.get('mutation_rate', 'N/A')} / {cfg.get('crossover_rate', 'N/A')}"),
        ("Maximum depth", cfg.get("max_depth", "N/A")), ("Horizon", f"{cfg.get('horizon', 'N/A')} days"),
    ]
    method_grid = "".join(f'<div><span>{esc(k)}</span><strong>{esc(v)}</strong></div>' for k,v in methodology)
    comparison_total = sum(int(row.get("evaluation_budget", 0)) for row in paired)
    dataset_hash = str(dataset.get("sha256", ""))[:16]

    # Summary charts use all available validation values and label missing bars as N/A.
    validation_chart_series = [
        ("Top-10 train |ICIR|", [evo_train_top, rnd_train_top], "#6785a1"),
        ("Direction-adjusted validation ICIR", [evo_valid_mean, rnd_valid_mean], "#a179b3"),
    ]
    # Represent aggregate categories with individual bars in a clear signed comparison.
    validation_compare_svg = chart_grouped(
        ["Evolution train", "Evolution validation", "Random train", "Random validation"],
        [("Mean ICIR", [evo_train_top, evo_valid_mean, rnd_train_top, rnd_valid_mean], "#087f78")],
        min_zero=False, decimals=3,
    )

    chart_roots = chart_grouped(root_categories,
        [(method["name"], method["values"], COLORS[EVOLUTION] if method["name"] == LABELS[EVOLUTION] else COLORS[RANDOM]) for method in root_series],
        decimals=1)
    root_legend = legend([(method["name"], [], method["color"]) for method in root_series])

    report_data = json.dumps(summary, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    title = "Random Search vs Evolutionary Search for Alpha Factor Discovery"
    subtitle = "Controlled five-seed comparison on CSI 300 · 20-day horizon"

    def table(headers: list[str], rows: list[str], cls: str = "") -> str:
        head = "".join(f"<th>{h}</th>" for h in headers)
        return f'<div class="table-scroll"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    html_report = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Five-seed paired comparison of random and evolutionary formulaic alpha search on CSI 300 data.">
<title>{esc(title)}</title><style>
:root{{--ink:#142e3a;--muted:#607783;--paper:#f3f6f5;--card:#fff;--line:#dce6e4;--teal:#087f78;--orange:#e18a48;--blue:#567fa6;--purple:#8c6ea6;--red:#b55357;--green:#237964;--shadow:0 12px 36px #1838450c}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}.page{{max-width:1440px;margin:auto;padding:28px 38px 70px}}header.hero{{background:linear-gradient(120deg,#153a49,#1f5960);color:#f5fbf9;border-radius:24px;padding:42px 46px;position:relative;overflow:hidden;box-shadow:var(--shadow)}}header.hero:after{{content:"";position:absolute;width:400px;height:400px;border:1px solid #ffffff20;border-radius:50%;right:-70px;top:-250px;box-shadow:0 0 0 45px #ffffff08,0 0 0 95px #ffffff06}}.eyebrow{{color:#a8ded2;text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:800}}h1{{font-size:clamp(32px,4vw,52px);line-height:1.05;letter-spacing:-.045em;max-width:850px;margin:13px 0}}.subtitle{{font-size:17px;color:#d5e7e4;margin:0 0 20px}}.hero-meta{{display:flex;flex-wrap:wrap;gap:8px}}.pill{{border:1px solid #ffffff32;background:#ffffff12;border-radius:30px;padding:6px 12px;font-size:11px;color:#e5f1ef}}.hero-note{{position:relative;z-index:1;margin-top:28px;max-width:880px;color:#deece9;font-size:13px}}nav{{position:sticky;top:0;z-index:10;background:#f3f6f5ee;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);margin:0 -38px 24px;padding:12px 38px;display:flex;gap:17px;overflow:auto}}nav a{{white-space:nowrap;color:#48636d;text-decoration:none;font-size:11px;font-weight:750}}nav a:hover{{color:var(--teal)}}section{{scroll-margin-top:66px;margin-top:38px}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:0 2px 14px}}h2{{font-size:23px;letter-spacing:-.03em;margin:0}}h3{{font-size:16px;letter-spacing:-.015em;margin:0 0 6px}}.section-desc,.subtle{{color:var(--muted);font-size:12px;margin:4px 0 0}}.cards{{display:grid;gap:14px}}.kpis{{grid-template-columns:repeat(4,minmax(0,1fr))}}.card{{background:var(--card);border:1px solid var(--line);border-radius:17px;box-shadow:var(--shadow)}}.kpi{{padding:21px;min-height:154px;position:relative;overflow:hidden}}.kpi:after{{content:"";position:absolute;width:90px;height:90px;border-radius:50%;right:-22px;bottom:-38px;background:var(--wash,#e6f3f0)}}.kpi-label{{color:var(--muted);font-weight:750;font-size:11px;text-transform:uppercase;letter-spacing:.08em}}.kpi-main{{font-size:29px;letter-spacing:-.045em;font-weight:820;margin:11px 0 2px}}.kpi-main small{{font-size:13px;font-weight:700;color:var(--muted)}}.kpi-note{{font-size:11px;color:var(--muted);max-width:260px}}.methodology{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;padding:8px 18px}}.methodology div{{padding:12px 13px;border-bottom:1px solid #edf1f0}}.methodology span,.methodology strong{{display:block}}.methodology span{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.07em}}.methodology strong{{font-size:12px;margin-top:3px}}.callout{{border-left:4px solid var(--teal);padding:18px 20px;background:#e8f3f0;border-radius:0 13px 13px 0;margin-top:16px}}.callout p{{margin:5px 0 0}}.warning{{border-left-color:var(--orange);background:#fbf1e7}}.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}}.chart-card{{padding:20px;min-width:0}}.chart-card.wide{{grid-column:1/-1}}.chart-title{{font-weight:780;font-size:14px}}.chart-caption{{font-size:11px;color:var(--muted);min-height:34px;margin:2px 0 6px}}.chart{{width:100%;height:auto;overflow:visible}}.gridline{{stroke:#e7edeb;stroke-width:1}}.zero-line{{stroke:#85989d;stroke-width:1.2;stroke-dasharray:4 4}}.axis{{fill:#607783;font-size:11px}}.x-label{{font-weight:680}}.bar-value{{fill:#344f59;font-size:10px;font-weight:750}}.bar{{transition:opacity .15s}}.bar:hover{{opacity:.76}}.na-label{{fill:#a3afb2;font-size:10px}}.legend{{display:flex;flex-wrap:wrap;gap:14px;margin:7px 0 0;font-size:10px;color:var(--muted)}}.legend span{{display:flex;align-items:center;gap:6px}}.legend i{{width:9px;height:9px;border-radius:3px;display:inline-block}}.sd-band{{fill:#087f781c;stroke:none}}.gen-line{{fill:none;stroke:var(--teal);stroke-width:3}}.errorbar{{stroke:#4b9b91;stroke-width:1.5}}.gen-point{{fill:#fff;stroke:var(--teal);stroke-width:3}}.table-scroll{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:#fff;box-shadow:var(--shadow)}}table{{width:100%;border-collapse:collapse;min-width:770px;font-size:12px}}th,td{{padding:11px 12px;border-bottom:1px solid #edf1f0;text-align:right;font-variant-numeric:tabular-nums;vertical-align:top}}thead th{{position:sticky;top:0;background:#f7faf9;color:#546c76;text-transform:uppercase;letter-spacing:.055em;font-size:9px;line-height:1.35;white-space:nowrap;cursor:pointer}}thead th:first-child,tbody th{{text-align:left}}tbody th{{font-weight:760;color:#294551;white-space:nowrap}}tbody tr:last-child td,tbody tr:last-child th{{border-bottom:0}}tbody tr:hover{{background:#f8fbfa}}td:first-child{{text-align:left}}.winner{{background:#e7f4ef;color:#126d5d;font-weight:800;border-radius:4px}}.directional{{font-weight:800;color:#654c76}}.n-cell{{color:#788990;text-align:left;font-size:10px;white-space:nowrap}}.method-cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.method-card{{padding:21px}}.method-label{{display:flex;align-items:center;gap:8px;font-weight:800}}.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}.method-card p{{color:var(--muted);font-size:12px}}.statline{{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding:8px 0;font-size:11px}}.statline strong{{font-variant-numeric:tabular-nums}}.verdict-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}}.verdict{{padding:16px}}.verdict b{{display:block;color:var(--teal);font-size:11px;text-transform:uppercase;letter-spacing:.06em}}.verdict p{{font-size:12px;margin:8px 0 0;color:#45606a}}.professor{{background:linear-gradient(115deg,#163b49,#174f54);color:#f4fbf9;padding:30px;border-radius:22px;margin-top:42px}}.professor h2{{font-size:26px}}.prof-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}.prof-panel{{background:#ffffff12;border:1px solid #ffffff20;padding:19px;border-radius:14px}}.prof-panel span{{display:block;color:#a8ded2;font-size:10px;font-weight:850;letter-spacing:.1em}}.prof-panel strong{{display:block;font-size:19px;margin:7px 0}}.professor p{{color:#d9e9e6;font-size:13px;max-width:1070px}}.footnote{{color:var(--muted);font-size:11px;margin-top:14px}}.tag{{display:inline-block;background:#edf3f2;border-radius:20px;padding:4px 9px;font-size:10px;color:#526b74}}.formula-cell code{{font:10px ui-monospace,monospace;overflow-wrap:anywhere;color:#33515b}}.empty{{color:var(--muted);padding:14px}}.nowrap{{white-space:nowrap}}.split-note{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}}.small-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}footer{{margin:28px 2px;color:var(--muted);font-size:10px}}@media(max-width:980px){{.page{{padding:18px 20px 48px}}nav{{margin:0 -20px 18px;padding:11px 20px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.methodology{{grid-template-columns:repeat(3,1fr)}}.verdict-grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:650px){{header.hero{{padding:28px 23px}}.page{{padding:14px 12px 40px}}nav{{margin:0 -12px 15px;padding:10px 12px}}.chart-grid,.method-cards,.small-grid{{grid-template-columns:1fr}}.chart-card.wide{{grid-column:auto}}.methodology{{grid-template-columns:repeat(2,1fr)}}.verdict-grid{{grid-template-columns:1fr 1fr}}.prof-grid{{grid-template-columns:1fr}}h2{{font-size:20px}}}}@media print{{body{{background:white}}.page{{max-width:none;padding:0 10mm}}nav{{position:static;display:none}}.card,.table-scroll{{box-shadow:none;break-inside:avoid}}header.hero,.professor{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}section{{break-inside:avoid-page}}thead th{{position:static}}}}
</style></head><body><div class="page">
<header class="hero"><div class="eyebrow">Aldien · Alpha Discovery Engine · Paired baseline study</div><h1>{esc(title)}</h1><p class="subtitle">{esc(subtitle)}</p><div class="hero-meta"><span class="pill">CSI 300 panel · {dataset.get('rows', 'N/A'):,} rows</span><span class="pill">{len(seeds)} seeds · {comparison_total:,} evaluations per method (sum of matched budgets)</span><span class="pill">Dataset SHA-256 · {esc(dataset_hash)}…</span><span class="pill">Validation used only for reporting</span></div><p class="hero-note">A controlled comparison of training optimization, expression-structure exploration, and the stability of selected factor directions on an unseen validation period.</p></header>
<nav><a href="#summary">Summary</a><a href="#training">Training fitness</a><a href="#generations">Convergence</a><a href="#validation">Validation</a><a href="#direction">Direction stability</a><a href="#diversity">Diversity</a><a href="#efficiency">Search cost</a><a href="#aggregate">Aggregates</a><a href="#verdict">Verdict</a></nav>
<section id="summary"><div class="section-head"><div><h2>Executive summary</h2><p class="section-desc">The central result is a trade-off: stronger training concentration from evolution, broader structures from random search, and weak directional persistence in validation.</p></div><span class="tag">Seed 67 validation unavailable for evolution</span></div><div class="cards kpis">
<article class="card kpi" style="--wash:#dff0eb"><div class="kpi-label">Training optimization · mean top-10 |ICIR|</div><div class="kpi-main">{fmt(evo_train_top)} <small>vs {fmt(rnd_train_top)}</small></div><div class="kpi-note">Evolutionary mean top-10 score is higher in {top_wins[0]}/{len(seeds)} seeds. Best-single-candidate average does not favor evolution.</div></article>
<article class="card kpi" style="--wash:#faeadb"><div class="kpi-label">Structural exploration · operator Jaccard</div><div class="kpi-main">{fmt(rnd_div)} <small>vs {fmt(evo_div)}</small></div><div class="kpi-note">Random’s top-10 formulas use a broader mix of operators. This is structural diversity, not proof of better alpha quality.</div></article>
<article class="card kpi" style="--wash:#eee7f3"><div class="kpi-label">Direction-adjusted validation · paired seeds</div><div class="kpi-main">{fmt(paired_direction[EVOLUTION])} <small>vs {fmt(paired_direction[RANDOM])}</small></div><div class="kpi-note">Means over the {len(paired_valid_seeds)} seeds with usable validation for both methods; random is less negative in {sum(m(s,RANDOM)['mean_validation_directional_icir_top_k'] > m(s,EVOLUTION)['mean_validation_directional_icir_top_k'] for s in paired_valid_seeds)}/{len(paired_valid_seeds)}.</div></article>
<article class="card kpi" style="--wash:#f7e7e5"><div class="kpi-label">Exploration–exploitation trade-off</div><div class="kpi-main">Train ↑<small> · diversity ↓</small></div><div class="kpi-note">Evolutionary search concentrates on high-training-score structures; possible premature convergence remains a hypothesis.</div></article></div>
<div class="callout"><strong>Interpretation in one sentence</strong><p>Evolutionary Search consistently improves the mean fitness of its training finalists, while Random Search preserves substantially broader operator structure; validation shows that training direction is not stable enough to claim an out-of-sample advantage.</p></div><div class="cards card methodology" style="margin-top:15px">{method_grid}</div></section>

<section id="training"><div class="section-head"><div><h2>1 · Training optimization</h2><p class="section-desc">Fitness is absolute training ICIR; bars compare each seed at the same evaluation budget.</p></div></div>{table(["Seed","Evolution · best |ICIR|","Random · best |ICIR|","Evolution · mean top-10","Random · mean top-10"], t1_rows)}<div class="chart-grid" style="margin-top:15px"><article class="card chart-card"><div class="chart-title">Best training absolute ICIR</div><p class="chart-caption">One best candidate per method and seed. Hover on bars for exact values.</p>{chart_grouped([str(s) for s in seeds], best_series)}{legend(best_series)}</article><article class="card chart-card"><div class="chart-title">Mean top-10 training absolute ICIR</div><p class="chart-caption">Average training fitness among the ten highest-ranked candidates.</p>{chart_grouped([str(s) for s in seeds], mean_series)}{legend(mean_series)}</article></div><div class="callout"><strong>What the training numbers support</strong><p>Evolution’s mean top-10 training |ICIR| is higher in all five seeds ({fmt(evo_train_top)} vs {fmt(rnd_train_top)} on average). The average best single-candidate score is {fmt(mean([m(s,EVOLUTION)['best_train_abs_icir'] for s in seeds]))} for evolution and {fmt(mean([m(s,RANDOM)['best_train_abs_icir'] for s in seeds]))} for random, so the best-candidate metric is mixed rather than an evolutionary win.</p></div></section>

<section id="generations"><div class="section-head"><div><h2>2 · Evolutionary training progression</h2><p class="section-desc">Mean best absolute training ICIR by generation across seeds; shaded region and whiskers show ± one sample standard deviation.</p></div><span class="tag">Generation 0 = initialized population</span></div><article class="card chart-card wide"><div class="chart-title">Best training fitness across generations</div><p class="chart-caption">All five evolutionary result files include 11 generation records (0 through 10).</p>{chart_generation(histories)}<div class="legend"><span><i style="background:#087f78"></i>Mean best |ICIR|</span><span><i style="background:#cfe8e3"></i>±1 standard deviation across seeds</span></div></article><div style="margin-top:14px">{table(["Generation","Mean best |ICIR|","Across-seed SD","Improvement vs Gen 0","Improvement (%)","Seeds"],generation_rows)}</div><p class="footnote">This chart describes optimization on the training split only. It does not demonstrate validation or investment performance.</p></section>

<section id="validation"><div class="section-head"><div><h2>3 · Training versus validation</h2><p class="section-desc">Each row summarizes that method’s top-10 train-ranked candidates. Direction-adjusted ICIR multiplies validation ICIR by the sign learned on training.</p></div></div>{table(["Seed","Method","Mean top-10 train |ICIR|","Best raw validation ICIR","Mean raw validation ICIR","Best direction-adjusted validation ICIR","Mean direction-adjusted validation ICIR","Train − validation |ICIR|"],t2_rows)}<div class="callout warning"><strong>Read the validation columns carefully</strong><p>Raw validation ICIR describes the validation association’s own sign. Direction-adjusted validation ICIR asks whether the training relationship persisted. The “Train − validation |ICIR|” column is an absolute-magnitude difference and ignores sign; a negative value does not establish stable generalization.</p></div><div class="chart-grid" style="margin-top:15px"><article class="card chart-card"><div class="chart-title">Mean training versus direction-adjusted validation</div><p class="chart-caption">Available validation means by method; negative validation values mean the training direction did not persist.</p>{validation_compare_svg}<div class="legend"><span><i style="background:#087f78"></i>Mean top-10 train |ICIR| / mean directional validation ICIR</span></div></article><article class="card chart-card"><div class="chart-title">Per-seed mean direction-adjusted validation ICIR</div><p class="chart-caption">All valid candidates are oriented using each formula’s training sign; unavailable runs remain N/A.</p>{chart_grouped([str(s) for s in seeds],[(LABELS[method],[m(s,method).get("mean_validation_directional_icir_top_k") for s in seeds],COLORS[method]) for method in METHODS],min_zero=False)}{legend([(LABELS[method],[],COLORS[method]) for method in METHODS])}</article></div></section>

<section id="direction"><div class="section-head"><div><h2>4 · Validation direction stability</h2><p class="section-desc">Sign agreement is recomputed from finite candidate-level validation ICIR values; it is not counted when a validation IC is undefined.</p></div></div>{table(["Seed","Method","Training ICIR signs · top-10","Validation ICIR signs · top-10","Direction agreement"],sign_rows)}<div class="chart-grid" style="margin-top:15px"><article class="card chart-card"><div class="chart-title">Candidate-level direction agreement by seed</div><p class="chart-caption">Fraction of top-10 formulas whose validation ICIR sign matches their training ICIR sign.</p>{chart_grouped([str(s) for s in seeds],direction_series,decimals=2)}{legend(direction_series)}</article><article class="card chart-card"><div class="chart-title">Aggregate direction agreement</div><p class="chart-caption">Method mean over runs with at least one finite validation ICIR.</p><div class="big-compare"><div><span>Evolution · {len(valid_seeds_by_method[EVOLUTION])}/{len(seeds)} runs</span><strong>{fmt(mean([v for v in sign_data[EVOLUTION] if finite(v)]),places=1)}</strong></div><div><span>Random · {len(valid_seeds_by_method[RANDOM])}/{len(seeds)} runs</span><strong>{fmt(mean([v for v in sign_data[RANDOM] if finite(v)]),places=1)}</strong></div></div><p class="footnote">In seed 67, the summary’s stored evolutionary agreement value is 0.0, but none of the ten finalists has a finite validation ICIR. This report correctly marks agreement as N/A for that run.</p></article></div><div class="callout"><strong>Validation is a diagnostic, not a method failure verdict</strong><p>Across usable runs, evolution’s higher training fitness did not translate into stable direction on validation. This motivates explicit coverage constraints and generalization controls before adding the LLM stage.</p></div></section>

<section id="diversity"><div class="section-head"><div><h2>5 · Structural diversity and convergence</h2><p class="section-desc">Operator-set Jaccard distance is a structural proxy: higher values mean less overlap in operators used, not lower correlation of alpha outputs.</p></div></div>{table(["Seed","Method","Root operators in top-10 · count","Mean pairwise operator Jaccard distance","Unique top-10 expressions"],diversity_rows)}<div class="chart-grid" style="margin-top:15px"><article class="card chart-card"><div class="chart-title">Mean pairwise operator Jaccard distance</div><p class="chart-caption">Higher means broader operator-set variety among the top ten expressions.</p>{chart_grouped([str(s) for s in seeds],diversity_series)}{legend(diversity_series)}</article><article class="card chart-card"><div class="chart-title">Root operators represented</div><p class="chart-caption">Average number of top-10 finalists per seed with each root operator; full count table above.</p>{chart_roots}{root_legend}</article></div><div class="callout"><strong>Exploration versus exploitation</strong><p>Random Search has higher operator-set diversity in every seed, while evolution’s top candidates cluster more tightly. This supports the interpretation that evolution is exploiting concentrated high-fitness regions; it is consistent with possible premature convergence, but does not by itself prove it.</p></div></section>

<section id="efficiency"><div class="section-head"><div><h2>6 · Search cost</h2><p class="section-desc">Both methods have matched unique evaluation budgets per seed. Runtime is shown as measured on this machine.</p></div></div>{table(["Seed","Method","Unique evaluations","Runtime (s)","Best training |ICIR|","Best raw validation ICIR"],cost_rows)}<div class="chart-grid" style="margin-top:15px"><article class="card chart-card"><div class="chart-title">Runtime by seed</div><p class="chart-caption">Includes train evaluation and finalist validation; not portable across hardware.</p>{chart_grouped([str(s) for s in seeds],runtime_series,suffix="s",decimals=0)}{legend(runtime_series)}</article><article class="card chart-card"><div class="chart-title">Efficiency metric decision</div><p class="subtle">No “ICIR per evaluation” ratio is plotted: dividing ICIR by candidate count does not have a standard interpretation and could mislead. The fair comparison is stronger training score at the same unique-evaluation budget, alongside observed runtime.</p><div class="big-compare"><div><span>Mean matched evaluations</span><strong>{fmt(mean([m(s,EVOLUTION).get('unique_evaluations') for s in seeds]),places=1)}</strong></div><div><span>Mean runtime · Evolution / Random</span><strong>{fmt(mean([m(s,EVOLUTION).get('elapsed_seconds') for s in seeds]),places=1)}s / {fmt(mean([m(s,RANDOM).get('elapsed_seconds') for s in seeds]),places=1)}s</strong></div></div></article></div></section>

<section id="aggregate"><div class="section-head"><div><h2>7 · Five-seed aggregate</h2><p class="section-desc">Means use finite values only. Validation rows show their usable sample counts; for the headline paired validation comparison use identical seeds only.</p></div></div>{table(["Metric","Evolutionary mean","n","Random mean","n","Reading"],aggregate_rows)}<div class="split-note"><span class="tag">Validation mean ICIR: Evolution {len(valid_seeds_by_method[EVOLUTION])}/{len(seeds)} · Random {len(valid_seeds_by_method[RANDOM])}/{len(seeds)}</span><span class="tag">Paired direction-adjusted validation: {len(paired_valid_seeds)}/{len(seeds)} common seeds</span><span class="tag">Evolution top-10 training mean wins: {top_wins[0]}/{len(seeds)}</span><span class="tag">Best single training candidate wins: Evolution {best_wins[0]} · Random {best_wins[1]}</span></div><p class="footnote">Raw signed validation ICIR can average formulas with opposite training directions and is not the generalization headline. Direction-adjusted validation ICIR fixes each formula’s sign from train. The stored train−validation magnitude gap does not account for that sign.</p></section>

<section id="verdict"><div class="section-head"><div><h2>8 · What the experiment tells us</h2><p class="section-desc">A defensible conclusion about the baseline stage—without claiming a validated trading strategy.</p></div></div><div class="verdict-grid"><article class="card verdict"><b>1 · Optimization</b><p>Evolutionary Search improves mean top-10 training fitness relative to random search in all five seeds.</p></article><article class="card verdict"><b>2 · Exploitation</b><p>Evolution’s top formulas cluster structurally, indicating concentrated search and possible premature convergence.</p></article><article class="card verdict"><b>3 · Exploration</b><p>Random Search samples a broader mix of expression structures; diversity is not equivalent to predictive quality.</p></article><article class="card verdict"><b>4 · Generalization</b><p>Direction-adjusted validation is negative for both methods on average; evolution does not show a validation advantage.</p></article><article class="card verdict"><b>5 · Next stage</b><p>Test whether LLM-guided semantic variation can add meaningful diversity while retaining fitness-driven selection.</p></article></div>
<div class="professor"><div class="eyebrow">Professor view · key experimental finding</div><h2>Optimization improved; generalization remains unresolved</h2><div class="prof-grid"><div class="prof-panel"><span>EXPLOITATION</span><strong>Evolutionary Search</strong><div>Higher mean top-10 training |ICIR|</div></div><div class="prof-panel"><span>EXPLORATION</span><strong>Random Search</strong><div>Higher structural diversity</div></div><div class="prof-panel"><span>NEXT STEP</span><strong>LLM-Guided Evolution</strong><div>Semantic exploration plus selection</div></div></div><p>In this matched-budget five-seed experiment, evolutionary search more consistently concentrates on higher-scoring training formulas, while random search retains broader operator diversity. The direction-adjusted validation results show that training relationships are not stable enough to establish an out-of-sample edge. The next experiment asks whether an LLM can introduce semantically meaningful exploration without losing evolutionary exploitation.</p></div></section>

<section><div class="section-head"><div><h2>Interpretation and limitations</h2><p class="section-desc">Use these qualifications when presenting the charts.</p></div></div><div class="small-grid"><article class="card chart-card"><h3>What is controlled</h3><p class="subtle">Same dataset SHA-256, train/validation splits, 20-day horizon, expanded seed pool, maximum depth 8, top-10 finalist count, and per-seed unique evaluation budget. Validation did not select the candidates.</p></article><article class="card chart-card"><h3>What is not established</h3><p class="subtle">Statistical significance after multiple testing, robustness across market regimes, net returns after turnover/transaction costs, or a live-tradable strategy. Test and holdout were not used in this comparison.</p></article><article class="card chart-card"><h3>Seed 67</h3><p class="subtle">All ten evolutionary finalists have zero valid validation IC days (reported coverage 0.0). The report shows validation fields as N/A and excludes the run from validation-agreement averages.</p></article><article class="card chart-card"><h3>Diversity definition</h3><p class="subtle">Jaccard distance compares sets of DSL operators in formulas. It is not pairwise alpha-value correlation, economic diversification, or a direct measurement of strategy risk.</p></article></div></section>
<footer>Source: paired comparison summary and its per-seed JSON results · Dataset {esc(dataset.get('path','N/A'))} · SHA-256 {esc(dataset.get('sha256','N/A'))} · This report is generated from recorded results; unavailable values are left as N/A.</footer>
</div><script>
// Table headers sort rows locally; all charts/data are embedded and need no network.
document.querySelectorAll('table').forEach(table=>{{table.querySelectorAll('thead th').forEach((th,index)=>{{th.addEventListener('click',()=>{{const body=table.tBodies[0];const rows=[...body.rows];const asc=th.dataset.asc!=='true';table.querySelectorAll('thead th').forEach(h=>h.dataset.asc='');th.dataset.asc=String(asc);rows.sort((a,b)=>{{const av=a.cells[index]?.innerText.trim()||'',bv=b.cells[index]?.innerText.trim()||'';const an=Number(av.replace(/[^0-9.+-]/g,'')),bn=Number(bv.replace(/[^0-9.+-]/g,''));const numeric=av!==''&&bv!==''&&Number.isFinite(an)&&Number.isFinite(bn);return (numeric?an-bn:av.localeCompare(bv,undefined,{{numeric:true}}))*(asc?1:-1)}});rows.forEach(r=>body.appendChild(r))}})}})}});
</script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="comparison_summary.json")
    parser.add_argument("--output", type=Path, required=True, help="destination HTML file")
    args = parser.parse_args()
    build(args.input, args.output)
    print(f"Presentation report: {args.output}")


if __name__ == "__main__":
    main()
