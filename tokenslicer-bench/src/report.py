"""Executive scorecard: DuckDB query -> rich table + results/scorecard.md + results/summary.json."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import PIPELINE_A, PIPELINE_B, PIPELINE_C, PIPELINE_ORDER
from .storage import ResultStore
from .tokens import tokenizer_name

log = logging.getLogger("tokenslicer.report")

LABELS = {
    PIPELINE_A: "A Naive Frontier",
    PIPELINE_B: "B Raw+Cheap",
    PIPELINE_C: "C TokenSlicer",
}

_METRICS_SQL = """
WITH ok AS (SELECT * FROM runs WHERE status = 'ok'),
pipes AS (SELECT COUNT(DISTINCT pipeline_name) AS n FROM ok),
paired AS (
    SELECT instance_id FROM ok GROUP BY instance_id
    HAVING COUNT(DISTINCT pipeline_name) = (SELECT n FROM pipes)
)
SELECT
    pipeline_name,
    COUNT(*)                                                    AS n_issues,
    SUM(cost_usd)                                               AS total_cost_usd,
    SUM(cost_inr)                                               AS total_cost_inr,
    AVG(cost_usd)                                               AS avg_cost_usd,
    AVG(latency_sec)                                            AS avg_latency_sec,
    SUM(input_tokens)                                           AS total_input_tokens,
    SUM(cached_input_tokens)                                    AS total_cached_input_tokens,
    SUM(output_tokens)                                          AS total_output_tokens,
    AVG(input_tokens)                                           AS avg_input_tokens,
    AVG(tokens_saved_pct)                                       AS avg_context_token_reduction_pct,
    100.0 * AVG(CAST(valid_git_diff AS DOUBLE))                 AS valid_patch_rate_pct,
    100.0 * AVG(CAST(gold_file_match AS DOUBLE))                AS gold_file_match_rate_pct,
    100.0 * AVG(CAST(gold_in_context AS DOUBLE))                AS gold_in_context_rate_pct,
    SUM(CASE WHEN usage_estimated THEN 1 ELSE 0 END)            AS rows_with_estimated_usage,
    SUM(CASE WHEN raw_truncated THEN 1 ELSE 0 END)              AS rows_raw_context_truncated
FROM ok
WHERE instance_id IN (SELECT instance_id FROM paired) OR NOT EXISTS (SELECT 1 FROM paired)
GROUP BY pipeline_name
"""


def _num(v: Any) -> float | None:
    return None if v is None else float(v)


def _pct_drop(new: float | None, base: float | None) -> float | None:
    if new is None or not base:
        return None
    return round((1 - new / base) * 100, 2)


def build_summary(store: ResultStore, usd_to_inr: float) -> dict[str, Any]:
    rows = store.query(_METRICS_SQL)
    by_name = {r["pipeline_name"]: r for r in rows}
    ordered = [by_name[p] for p in PIPELINE_ORDER if p in by_name] + [
        r for r in rows if r["pipeline_name"] not in PIPELINE_ORDER
    ]

    errors = {
        r["pipeline_name"]: int(r["n"])
        for r in store.query("SELECT pipeline_name, COUNT(*) AS n FROM runs WHERE status <> 'ok' GROUP BY 1")
    }
    n_issues_total = store.query("SELECT COUNT(DISTINCT instance_id) AS n FROM runs")[0]["n"]

    pipelines: list[dict[str, Any]] = []
    for r in ordered:
        pipelines.append(
            {
                "pipeline": r["pipeline_name"],
                "label": LABELS.get(r["pipeline_name"], r["pipeline_name"]),
                "issues_scored": int(r["n_issues"]),
                "errors_excluded": errors.get(r["pipeline_name"], 0),
                "total_cost_usd": round(float(r["total_cost_usd"] or 0), 6),
                "total_cost_inr": round(float(r["total_cost_inr"] or 0), 4),
                "avg_cost_usd": round(float(r["avg_cost_usd"] or 0), 6),
                "avg_latency_sec": round(float(r["avg_latency_sec"] or 0), 3),
                "total_input_tokens": int(r["total_input_tokens"] or 0),
                "total_cached_input_tokens": int(r["total_cached_input_tokens"] or 0),
                "total_output_tokens": int(r["total_output_tokens"] or 0),
                "avg_input_tokens": round(float(r["avg_input_tokens"] or 0), 1),
                "avg_context_token_reduction_pct": round(float(r["avg_context_token_reduction_pct"] or 0), 2),
                "valid_patch_rate_pct": round(float(r["valid_patch_rate_pct"] or 0), 2),
                "gold_file_match_rate_pct": None if r["gold_file_match_rate_pct"] is None else round(float(r["gold_file_match_rate_pct"]), 2),
                "gold_in_context_rate_pct": None if r["gold_in_context_rate_pct"] is None else round(float(r["gold_in_context_rate_pct"]), 2),
                "rows_with_estimated_usage": int(r["rows_with_estimated_usage"] or 0),
                "rows_raw_context_truncated": int(r["rows_raw_context_truncated"] or 0),
            }
        )

    idx = {p["pipeline"]: p for p in pipelines}
    a, b, c = idx.get(PIPELINE_A), idx.get(PIPELINE_B), idx.get(PIPELINE_C)
    headline = {
        "cost_reduction_c_vs_a_pct": _pct_drop(c and c["total_cost_usd"], a and a["total_cost_usd"]),
        "cost_reduction_c_vs_b_pct": _pct_drop(c and c["total_cost_usd"], b and b["total_cost_usd"]),
        "cost_reduction_b_vs_a_pct": _pct_drop(b and b["total_cost_usd"], a and a["total_cost_usd"]),
        "api_input_token_reduction_c_vs_b_pct": _pct_drop(c and c["total_input_tokens"], b and b["total_input_tokens"]),
        "valid_patch_rate_delta_c_minus_b_pts": (
            round(c["valid_patch_rate_pct"] - b["valid_patch_rate_pct"], 2) if b and c else None
        ),
        "latency_reduction_c_vs_b_pct": _pct_drop(c and c["avg_latency_sec"], b and b["avg_latency_sec"]),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usd_to_inr": usd_to_inr,
        "tokenizer": tokenizer_name(),
        "issues_in_db": int(n_issues_total),
        "issues_in_comparison": max((p["issues_scored"] for p in pipelines), default=0),
        "pipelines": pipelines,
        "headline": headline,
    }


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _usd(v: float) -> str:
    return f"${v:,.4f}"


def _inr(v: float) -> str:
    return f"₹{v:,.2f}"


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def render_console(summary: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    if not summary["pipelines"]:
        console.print("[bold red]No completed runs found in the database yet.[/]")
        return

    table = Table(
        title=f"TokenSlicer-Bench Scorecard  ({summary['issues_in_comparison']} issues, USD→INR {summary['usd_to_inr']})",
        box=box.HEAVY_HEAD,
        padding=(0, 1),
        header_style="bold white on blue",
        title_style="bold bright_white",
        caption="USD/INR=total cost | In Tok=avg billed input | Saved=context token reduction | Valid=git apply --check | Gold=touches human-fixed file",
        caption_style="dim",
    )
    table.add_column("Pipeline", style="bold bright_white", overflow="fold")
    for name in ("USD", "INR", "Latency", "In Tok", "Saved", "Valid", "Gold"):
        table.add_column(name, justify="right", no_wrap=True)

    for p in summary["pipelines"]:
        is_c = p["pipeline"] == PIPELINE_C
        style = "bold bright_green" if is_c else ("bright_red" if p["pipeline"] == PIPELINE_A else "bright_yellow")
        table.add_row(
            p["label"],
            _usd(p["total_cost_usd"]),
            _inr(p["total_cost_inr"]),
            f"{p['avg_latency_sec']:.1f}s",
            f"{p['avg_input_tokens']:,.0f}",
            _pct(p["avg_context_token_reduction_pct"]),
            _pct(p["valid_patch_rate_pct"]),
            _pct(p["gold_file_match_rate_pct"]),
            style=style,
        )
    console.print(table)

    h = summary["headline"]
    lines = []
    if h["cost_reduction_c_vs_a_pct"] is not None:
        lines.append(f"[bold bright_green]TokenSlicer (C) is {h['cost_reduction_c_vs_a_pct']:.1f}% cheaper than the naive frontier baseline (A)[/]")
    if h["cost_reduction_c_vs_b_pct"] is not None:
        lines.append(f"…and {h['cost_reduction_c_vs_b_pct']:.1f}% cheaper than sending raw context to the same cheap model (B)")
    if h["api_input_token_reduction_c_vs_b_pct"] is not None:
        lines.append(f"Billed input tokens C vs B: -{h['api_input_token_reduction_c_vs_b_pct']:.1f}%")
    if h["valid_patch_rate_delta_c_minus_b_pts"] is not None:
        lines.append(f"Valid-patch rate C vs B: {h['valid_patch_rate_delta_c_minus_b_pts']:+.1f} pts")
    if lines:
        console.print(Panel("\n".join(lines), title="Executive Summary", border_style="bright_green", padding=(1, 2)))
    for p in summary["pipelines"]:
        if p["errors_excluded"]:
            console.print(f"[yellow]{p['label']}: {p['errors_excluded']} errored row(s) excluded; re-run to retry them.[/]")


def render_markdown(summary: dict[str, Any]) -> str:
    h = summary["headline"]
    out: list[str] = [
        "# TokenSlicer-Bench Scorecard",
        "",
        f"_Generated {summary['generated_at']} · {summary['issues_in_comparison']} issues compared · "
        f"USD→INR {summary['usd_to_inr']} · token counter `{summary['tokenizer']}`_",
        "",
        "## Headline",
        "",
    ]
    if h["cost_reduction_c_vs_a_pct"] is not None:
        out.append(f"- **TokenSlicer (C) vs naive frontier (A): {h['cost_reduction_c_vs_a_pct']:.1f}% lower total cost**")
    if h["cost_reduction_c_vs_b_pct"] is not None:
        out.append(f"- C vs raw context on the same cheap model (B): {h['cost_reduction_c_vs_b_pct']:.1f}% lower total cost")
    if h["cost_reduction_b_vs_a_pct"] is not None:
        out.append(f"- B vs A (model arbitrage alone): {h['cost_reduction_b_vs_a_pct']:.1f}% lower total cost")
    if h["api_input_token_reduction_c_vs_b_pct"] is not None:
        out.append(f"- Billed input tokens, C vs B: -{h['api_input_token_reduction_c_vs_b_pct']:.1f}%")
    if h["latency_reduction_c_vs_b_pct"] is not None:
        lat = h["latency_reduction_c_vs_b_pct"]
        word = "faster" if lat >= 0 else "slower"
        out.append(f"- Avg latency, C vs B: {abs(lat):.1f}% {word} (includes AST-pruning time)")
    if h["valid_patch_rate_delta_c_minus_b_pts"] is not None:
        out.append(f"- Valid-patch rate, C minus B: {h['valid_patch_rate_delta_c_minus_b_pts']:+.1f} percentage points")
    out += [
        "",
        "## Comparison",
        "",
        "| Pipeline | Issues | Total cost (USD) | Total cost (INR) | Avg latency | Avg input tokens | Ctx token reduction | Valid patch rate | Gold-file match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in summary["pipelines"]:
        out.append(
            f"| {p['label']} | {p['issues_scored']} | {_usd(p['total_cost_usd'])} | {_inr(p['total_cost_inr'])} | "
            f"{p['avg_latency_sec']:.1f}s | {p['avg_input_tokens']:,.0f} | {_pct(p['avg_context_token_reduction_pct'])} | "
            f"{_pct(p['valid_patch_rate_pct'])} | {_pct(p['gold_file_match_rate_pct'])} |"
        )
    out += [
        "",
        "## How to read this",
        "",
        "- **A** re-prices B's real token counts at frontier rates when `PIPELINE_A_MODE=simulate` (default); its patch, latency and validity are therefore B's, not a real frontier model's.",
        "- **Ctx token reduction** is measured locally (raw vs pruned context) and is 0% for A and B by definition. **Avg input tokens** are what the provider reported.",
        "- **Valid patch rate** only means `git apply --check` succeeds. It does **not** mean the patch fixes the bug; no tests are executed.",
        "- **Gold-file match** = the patch touches at least one file changed by the human fix. A cheap proxy for localisation, not for correctness.",
        "- Only issues completed by every pipeline are compared (paired set). Errored rows are excluded and listed below.",
        "",
    ]
    errs = [p for p in summary["pipelines"] if p["errors_excluded"] or p["rows_raw_context_truncated"] or p["rows_with_estimated_usage"]]
    if errs:
        out += ["## Data-quality notes", ""]
        for p in errs:
            out.append(
                f"- {p['label']}: {p['errors_excluded']} errored rows, "
                f"{p['rows_raw_context_truncated']} raw contexts truncated to the token cap, "
                f"{p['rows_with_estimated_usage']} rows with locally-estimated usage"
            )
        out.append("")
    return "\n".join(out)


def write_artifacts(summary: dict[str, Any], results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / "scorecard.md"
    json_path = results_dir / "summary.json"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return md_path, json_path


def generate_report(store: ResultStore, results_dir: Path, usd_to_inr: float, console: Console | None = None) -> dict[str, Any]:
    summary = build_summary(store, usd_to_inr)
    render_console(summary, console)
    if summary["pipelines"]:
        md, js = write_artifacts(summary, results_dir)
        (console or Console()).print(f"[dim]Wrote {md} and {js}[/]")
    return summary
