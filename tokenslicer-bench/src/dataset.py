"""Issue loading: Hugging Face datasets or a local JSONL file."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from .models import Issue

log = logging.getLogger("tokenslicer.dataset")


class DatasetError(RuntimeError):
    pass


def load_issues_file(path: Path) -> list[Issue]:
    """JSONL, one object per line. Required: instance_id, problem_statement.
    Optional: repo, base_commit, patch (or gold_patch)."""
    if not path.is_file():
        raise DatasetError(f"Issues file not found: {path}")
    issues: list[Issue] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            issues.append(
                Issue(
                    instance_id=obj["instance_id"],
                    repo=obj.get("repo", ""),
                    problem_statement=obj["problem_statement"],
                    base_commit=obj.get("base_commit"),
                    gold_patch=obj.get("patch") or obj.get("gold_patch"),
                )
            )
        except (json.JSONDecodeError, KeyError, ValidationError) as exc:
            log.warning("skipping malformed issue line", extra={"file": str(path), "line": lineno, "error": str(exc)})
    return issues


def load_hf_issues(dataset: str, split: str, repo_filter: str) -> list[Issue]:
    from datasets import load_dataset  # imported lazily: heavy

    ds = load_dataset(dataset, split=split)
    needle = repo_filter.lower()
    rows = [
        r
        for r in ds
        if needle in str(r.get("repo", "")).lower() or needle in str(r.get("instance_id", "")).lower()
    ]
    if not rows:
        repos = sorted({str(r.get("repo", "")) for r in ds})
        raise DatasetError(
            f"No instances in {dataset}[{split}] match repo filter {repo_filter!r}.\n"
            f"Repos present in this dataset: {', '.join(repos)}\n"
            "SWE-bench / SWE-bench_Lite do not include apache/airflow. Either pick one of the repos above "
            "(clone it and pass --repo-filter/--repo-path), or build real Airflow issues with "
            "`python scripts/mine_airflow_issues.py` and pass --issues-file data/airflow_issues.jsonl."
        )
    issues = [
        Issue(
            instance_id=r["instance_id"],
            repo=r.get("repo", ""),
            problem_statement=r["problem_statement"],
            base_commit=r.get("base_commit"),
            gold_patch=r.get("patch"),
        )
        for r in rows
    ]
    return issues


def load_issues(
    *,
    dataset: str,
    split: str,
    repo_filter: str,
    issues_file: Path | None,
    limit: int | None,
) -> list[Issue]:
    issues = load_issues_file(issues_file) if issues_file else load_hf_issues(dataset, split, repo_filter)
    issues.sort(key=lambda i: i.instance_id)  # deterministic order => stable checkpoints and --limit
    if limit is not None:
        issues = issues[:limit]
    if not issues:
        raise DatasetError("No issues loaded.")
    return issues
