#!/usr/bin/env python3
"""TokenSlicer-Bench CLI.

    python run_benchmark.py --repo-path ./airflow --issues-file data/airflow_issues.jsonl --limit 25 --concurrency 5
    python run_benchmark.py --repo-path ./airflow --issues-file data/airflow_issues.jsonl --dry-run
    python run_benchmark.py --report-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import ConfigError, Settings
from src.context_pruner import ContextPruner
from src.dataset import DatasetError, load_issues
from src.logging_setup import setup_logging
from src.models import PIPELINE_LABELS, RunRecord
from src.report import generate_report
from src.router_engine import BenchmarkRunner
from src.storage import ResultStore

console = Console()
log = logging.getLogger("tokenslicer.cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TokenSlicer-Bench: LLM coding cost-arbitrage benchmark")
    p.add_argument("--repo-path", type=Path, default=Path("./airflow"), help="Local git clone of the target repo")
    p.add_argument("--limit", type=int, default=25, help="Max issues to run (deterministic order by instance_id)")
    p.add_argument("--concurrency", type=int, default=5, help="Max concurrent issues / API requests")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite", help="Hugging Face dataset id")
    p.add_argument("--split", default="test")
    p.add_argument("--repo-filter", default="airflow", help="Substring matched against the dataset's repo/instance_id")
    p.add_argument("--issues-file", type=Path, default=None, help="Local JSONL of issues (overrides --dataset)")
    p.add_argument("--pipelines", default="A,B,C", help="Comma list from A,B,C")
    p.add_argument("--top-k", type=int, default=5, help="Files retrieved per issue")
    p.add_argument("--context-lines", type=int, default=15, help="Lines of context around kept functions")
    p.add_argument("--max-functions", type=int, default=5, help="Max relevant functions kept per file (lower = more compression)")
    p.add_argument("--no-sibling-signatures", action="store_true",
                   help="Drop the one-line signatures of non-relevant methods in kept classes (more compression, less API visibility)")
    p.add_argument("--b-layout", choices=["naive", "cached"], default="naive",
                   help="Prompt layout for Pipeline B. 'cached' gives B the same cache-friendly layout as C, "
                        "isolating the effect of pruning alone")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--usd-inr", type=float, default=None, help="Override USD_TO_INR from .env")
    p.add_argument("--dry-run", action="store_true", help="Build contexts and show token reduction; no API calls, no DB writes")
    p.add_argument("--report-only", action="store_true", help="Skip running; just render the scorecard from the DB")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def resolve_pipelines(spec: str) -> list[str]:
    names: list[str] = []
    for token in spec.upper().replace(" ", "").split(","):
        if not token:
            continue
        if token not in PIPELINE_LABELS:
            raise SystemExit(f"Unknown pipeline {token!r}; choose from A,B,C")
        names.append(PIPELINE_LABELS[token])
    if not names:
        raise SystemExit("No pipelines selected")
    return names


def dry_run(pruner: ContextPruner, issues: list) -> None:
    table = Table(title="Dry run: context compression (no API calls)", header_style="bold white on blue")
    for col in ("Instance", "Files", "Raw tok", "Pruned tok", "Reduction", "Truncated"):
        table.add_column(col, no_wrap=True)
    table.add_column("Retrieved files", overflow="fold", min_width=36)
    total_raw = total_pruned = 0
    for issue in issues:
        b = pruner.build(issue.instance_id, issue.problem_statement)
        total_raw += b.raw_tokens
        total_pruned += b.pruned_tokens
        table.add_row(
            issue.instance_id, str(len(b.files)), f"{b.raw_tokens:,}", f"{b.pruned_tokens:,}",
            f"{b.reduction_pct:.1f}%", "yes" if b.raw_truncated else "", "\n".join(b.retrieved_paths),
        )
    console.print(table)
    if total_raw:
        console.print(f"[bold green]Aggregate context reduction: {(1 - total_pruned / total_raw) * 100:.1f}%[/] "
                      f"({total_raw:,} → {total_pruned:,} tokens, tokenizer: {tokenizer_label()})")


def tokenizer_label() -> str:
    from src.tokens import tokenizer_name
    return tokenizer_name()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir: Path = args.results_dir
    setup_logging(results_dir / "benchmark.log.jsonl", verbose=args.verbose)

    try:
        settings = Settings()
    except ValidationError as exc:
        console.print(f"[bold red]Invalid configuration in .env:[/]\n{exc}")
        return 2
    if args.usd_inr is not None:
        settings = settings.model_copy(update={"usd_to_inr": args.usd_inr})

    db_path = results_dir / "benchmark.duckdb"

    if args.report_only:
        with ResultStore(db_path) as store:
            generate_report(store, results_dir, settings.usd_to_inr, console)
        return 0

    pipelines = resolve_pipelines(args.pipelines)

    try:
        issues = load_issues(
            dataset=args.dataset, split=args.split, repo_filter=args.repo_filter,
            issues_file=args.issues_file, limit=args.limit,
        )
    except DatasetError as exc:
        console.print(f"[bold red]Dataset error:[/] {exc}")
        return 2
    console.print(f"Loaded [bold]{len(issues)}[/] issues.")

    try:
        pruner = ContextPruner(
            args.repo_path, top_k=args.top_k, context_lines=args.context_lines,
            max_functions_per_file=args.max_functions, sibling_signatures=not args.no_sibling_signatures,
            raw_token_cap=settings.raw_context_token_cap,
        )
        with console.status(f"Indexing {args.repo_path} with BM25 (once per run)..."):
            index = pruner.warm()
        console.print(f"Indexed [bold]{len(index.files):,}[/] Python files.")
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/]\nClone the repo first: git clone --depth 1 https://github.com/apache/airflow.git airflow")
        return 2

    if args.dry_run:
        dry_run(pruner, issues)
        return 0

    store = ResultStore(db_path)
    progress = Progress(
        SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
        TimeElapsedColumn(), console=console,
    )
    completed_before = store.completed_pairs()

    def on_record(rec: RunRecord) -> None:
        progress.advance(task_id)

    try:
        runner = BenchmarkRunner(
            settings=settings, store=store, pruner=pruner, repo_path=args.repo_path,
            pipelines=pipelines, concurrency=args.concurrency, b_layout=args.b_layout, on_record=on_record,
        )
    except ConfigError as exc:
        store.close()
        console.print(f"[bold red]Configuration error:[/] {exc}")
        return 2

    todo_total = sum(1 for i in issues for p in runner.pipelines if (i.instance_id, p) not in completed_before)
    skipped = len(issues) * len(runner.pipelines) - todo_total
    if skipped:
        console.print(f"[cyan]Resuming: {skipped} (issue, pipeline) results already checkpointed; skipping them.[/]")

    interrupted = False
    try:
        with progress:
            task_id = progress.add_task("Benchmarking", total=max(todo_total, 1))
            if todo_total == 0:
                progress.update(task_id, completed=1)
            stats = asyncio.run(runner.run(issues))
        console.print(
            f"Run complete: [green]{stats.pairs_ok} ok[/], [red]{stats.pairs_error} errors[/], {stats.pairs_skipped} skipped."
        )
        for line in stats.errors[:10]:
            console.print(f"  [red]•[/] {line}")
        if len(stats.errors) > 10:
            console.print(f"  … and {len(stats.errors) - 10} more (see results/benchmark.log.jsonl)")
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[yellow]Interrupted. All finished results are checkpointed; re-run the same command to resume.[/]")
    finally:
        # Report whatever is in the DB, even after Ctrl-C.
        generate_report(store, results_dir, settings.usd_to_inr, console)
        store.close()
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
