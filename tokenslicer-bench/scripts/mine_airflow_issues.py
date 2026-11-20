#!/usr/bin/env python3
"""Mine real Airflow bug-fix instances from GitHub into a SWE-bench-style JSONL.

SWE-bench and SWE-bench_Lite contain no apache/airflow instances, so this builds them:
merged PRs whose body says "closes: #N" -> (issue text = problem_statement, PR diff minus tests = gold patch).

    export GITHUB_TOKEN=ghp_...        # strongly recommended: 60 req/h unauthenticated vs 5000/h
    python scripts/mine_airflow_issues.py --out data/airflow_issues.jsonl --count 60

Caveats: `base_commit` is the PR's base-branch SHA at last update (approximate). With a shallow clone
at HEAD, file contents may differ from what the issue author saw. The benchmark measures cost and
patch applicability against the clone you provide, not historical resolve rate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

API = "https://api.github.com"
REPO = "apache/airflow"
CLOSES_RE = re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*(?:apache/airflow)?#(\d+)", re.I)
TEST_PATH_RE = re.compile(r"(^|/)(tests?|docs?|helm_tests|kubernetes_tests|system_tests)/|(^|/)(test_[^/]*|conftest)\.py$")


class RateLimited(Exception):
    pass


def make_client(token: str | None) -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "tokenslicer-bench"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=API, headers=headers, timeout=30.0, follow_redirects=True)


@retry(retry=retry_if_exception_type((RateLimited, httpx.TransportError)), wait=wait_exponential(min=5, max=120), stop=stop_after_attempt(5), reraise=True)
def get(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    resp = client.get(url, **kwargs)
    if resp.status_code in (403, 429) and resp.headers.get("x-ratelimit-remaining") == "0":
        reset = int(resp.headers.get("x-ratelimit-reset", "0"))
        wait = max(0, min(reset - int(time.time()), 120))
        print(f"rate limited; sleeping {wait}s", file=sys.stderr)
        time.sleep(wait)
        raise RateLimited(url)
    return resp


def split_diff(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into (path, section) pairs."""
    sections: list[tuple[str, str]] = []
    for chunk in re.split(r"(?m)^(?=diff --git )", diff):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
        if m:
            sections.append((m.group(2), chunk))
    return sections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/airflow_issues.jsonl"))
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--max-files", type=int, default=3, help="Skip PRs whose non-test .py changes touch more files than this")
    ap.add_argument("--min-issue-chars", type=int, default=300)
    args = ap.parse_args()

    client = make_client(os.environ.get("GITHUB_TOKEN"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(l)["instance_id"] for l in args.out.read_text().splitlines() if l.strip()} if args.out.exists() else set()
    written = 0

    page = 1
    query = f"repo:{REPO} is:pr is:merged label:\"kind:bug\" \"closes\" in:body"
    with args.out.open("a", encoding="utf-8") as fh:
        while written < args.count:
            resp = get(client, "/search/issues", params={"q": query, "sort": "updated", "order": "desc", "per_page": 50, "page": page})
            if resp.status_code != 200:
                print(f"search failed: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
                return 1
            items = resp.json().get("items", [])
            if not items:
                break
            for pr in items:
                if written >= args.count:
                    break
                number = pr["number"]
                instance_id = f"apache__airflow-{number}"
                if instance_id in seen:
                    continue
                linked = CLOSES_RE.findall(pr.get("body") or "")
                if not linked:
                    continue
                issue_resp = get(client, f"/repos/{REPO}/issues/{linked[0]}")
                if issue_resp.status_code != 200:
                    continue
                issue = issue_resp.json()
                if "pull_request" in issue:
                    continue
                statement = f"{issue['title']}\n\n{issue.get('body') or ''}".strip()
                if len(statement) < args.min_issue_chars:
                    continue

                pr_json = get(client, f"/repos/{REPO}/pulls/{number}").json()
                diff_resp = get(client, f"/repos/{REPO}/pulls/{number}", headers={"Accept": "application/vnd.github.v3.diff"})
                if diff_resp.status_code != 200:
                    continue
                src = [(p, s) for p, s in split_diff(diff_resp.text) if p.endswith(".py") and not TEST_PATH_RE.search(p)]
                if not src or len(src) > args.max_files:
                    continue

                row = {
                    "instance_id": instance_id,
                    "repo": REPO,
                    "base_commit": pr_json.get("base", {}).get("sha"),
                    "problem_statement": statement,
                    "patch": "".join(s for _p, s in src),
                    "pull_number": number,
                    "issue_number": issue["number"],
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                written += 1
                print(f"[{written}/{args.count}] {instance_id} <- issue #{issue['number']}", file=sys.stderr)
            page += 1
            if page > 20:
                break
    print(f"Wrote {written} new instances to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
