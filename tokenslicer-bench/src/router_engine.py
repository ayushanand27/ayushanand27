"""Async multi-model benchmark runner.

Pipelines (identical issues, identical retrieval):
  A  Naive Frontier Simulation   raw context, priced at frontier rates.
                                 simulate: re-price Pipeline B's real response (no frontier spend).
                                 live:     really call FRONTIER_* endpoint.
  B  Raw context + cheap worker  raw context, real usage, real worker pricing (naive prompt layout).
  C  TokenSlicer + cheap worker  AST-pruned context, cache-optimised prompt layout, real usage/pricing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import ProviderConfig, Settings
from .context_pruner import ContextPruner
from .models import (
    PIPELINE_A,
    PIPELINE_B,
    PIPELINE_C,
    ContextBundle,
    Issue,
    RunRecord,
)
from .storage import ResultStore
from .tokens import count_tokens

log = logging.getLogger("tokenslicer.router")

Layout = Literal["naive", "cached"]

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

ROLE = (
    "You are a senior software engineer fixing a bug in a large Python repository. "
    "You are given the issue and the most relevant source files (some files may be shown as a pruned view)."
)

FORMAT_RULES = """\
OUTPUT FORMAT (strict - your reply is fed directly to `git apply`):
1. Reply with ONE unified diff and nothing else: no explanations, no markdown fences, no commentary.
2. Every file section starts with `--- a/<path>` and `+++ b/<path>`, using the repo-relative paths shown in the `### FILE:` headers.
3. Every hunk starts with `@@ -<old_start>,<old_count> +<new_start>,<new_count> @@` and the counts MUST match the hunk body.
4. Include at least 3 unchanged context lines before and after each change. Copy context lines EXACTLY, including indentation and trailing text.
5. Lines starting with `# ... [lines N-M pruned` are elision markers, not real code. Never use them as context lines and never modify them. Use real neighbouring code for context.
6. Only modify files that appear in the provided context. Make the smallest change that fixes the issue. Do not touch tests unless the issue requires it.
7. If you cannot determine a fix, reply with an empty response.

EXAMPLE OF A VALID REPLY:
--- a/pkg/utils.py
+++ b/pkg/utils.py
@@ -10,7 +10,7 @@ def normalize(name):
     if name is None:
         return ""
     name = name.strip()
-    return name.lower
+    return name.lower()


 def join(parts):
"""

NAIVE_SYSTEM = "You are a helpful coding assistant."


def build_messages(context: str, problem_statement: str, layout: Layout) -> list[dict[str, str]]:
    """`cached`: a byte-identical static prefix (role + rules) comes first and the per-issue payload last,
    so providers with automatic prefix caching can reuse it across issues.
    `naive`: short system prompt, per-issue text first, instructions last (nothing shared across issues)."""
    if layout == "cached":
        return [
            {"role": "system", "content": f"{ROLE}\n\n{FORMAT_RULES}"},
            {
                "role": "user",
                "content": (
                    f"<repository_context>\n{context}\n</repository_context>\n\n"
                    f"<issue>\n{problem_statement.strip()}\n</issue>\n\n"
                    "Reply with the unified diff only."
                ),
            },
        ]
    return [
        {"role": "system", "content": NAIVE_SYSTEM},
        {
            "role": "user",
            "content": f"ISSUE:\n{problem_statement.strip()}\n\nSOURCE FILES:\n{context}\n\n{ROLE}\n\n{FORMAT_RULES}",
        },
    ]


# --------------------------------------------------------------------------------------
# Patch extraction + validation
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)(?:```|\Z)", re.S)
_DIFF_START_RE = re.compile(r"^(?:diff --git |--- (?:a/|/dev/null))", re.M)
_DIFF_LINE_PREFIXES = (" ", "+", "-", "@@", "\\", "diff ", "index ", "new file", "deleted file",
                       "similarity", "rename ", "old mode", "new mode", "Binary files")
_FILE_HDR_RE = re.compile(r"^(?:---|\+\+\+) (?:[ab]/)?(\S+)", re.M)


def extract_patch(text: str | None) -> str:
    """Pull a unified diff out of a model reply (tolerates fences and surrounding prose)."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    blocks = [
        b.rstrip("\n") + "\n"
        for b in _FENCE_RE.findall(text)
        if re.search(r"^--- ", b, re.M) and re.search(r"^\+\+\+ ", b, re.M)
    ]
    candidate = "".join(blocks) if blocks else text
    m = _DIFF_START_RE.search(candidate)
    if not m:
        return ""
    lines = candidate[m.start():].split("\n")
    while lines and (not lines[-1] or not lines[-1].startswith(_DIFF_LINE_PREFIXES)):
        lines.pop()
    return ("\n".join(lines) + "\n") if lines else ""


def diff_files(patch: str | None) -> set[str]:
    """Repo-relative paths touched by a unified diff."""
    if not patch:
        return set()
    return {p for p in _FILE_HDR_RE.findall(patch) if p != "/dev/null"}


async def git_apply_check(repo: Path, patch: str, *, recount: bool, timeout: float = 60.0) -> tuple[bool, str | None]:
    """`git apply --check` against the working tree of `repo` (never modifies it)."""
    if not patch.strip():
        return False, "empty patch"
    args = ["git", "apply", "--check", "--whitespace=nowarn"]
    if recount:
        args.append("--recount")
    args.append("-")
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(repo),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _out, err = await asyncio.wait_for(proc.communicate(patch.encode("utf-8")), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return False, f"git apply timed out after {timeout:.0f}s"
    except (OSError, ValueError) as exc:
        return False, f"git apply could not run: {exc}"
    if proc.returncode == 0:
        return True, None
    return False, (err.decode("utf-8", "replace").strip() or f"exit {proc.returncode}")[:500]


# --------------------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------------------

_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,  # includes APITimeoutError
    openai.InternalServerError,
)


@dataclass(slots=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    usage_estimated: bool
    latency_sec: float  # wall time of the successful attempt only (retry back-off excluded)
    finish_reason: str | None = None


def price_usd(cfg: ProviderConfig, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    cached = min(max(cached_tokens, 0), prompt_tokens)
    miss = prompt_tokens - cached
    return (miss * cfg.price_in_miss + cached * cfg.price_in_hit + completion_tokens * cfg.price_out) / 1_000_000


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class LLMClient:
    """OpenAI-compatible async client with bounded concurrency and exponential-backoff retries."""

    def __init__(self, cfg: ProviderConfig, settings: Settings, concurrency: int) -> None:
        self.cfg = cfg
        self.settings = settings
        self._sem = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_sec, connect=15.0),
            limits=httpx.Limits(max_connections=concurrency * 2 + 2, max_keepalive_connections=concurrency + 2),
        )
        self._client = AsyncOpenAI(
            base_url=cfg.base_url, api_key=cfg.api_key, http_client=self._http, max_retries=0
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0.0
        log.warning(
            "retrying after transient API error",
            extra={"attempt": state.attempt_number, "wait_sec": round(wait, 1), "error": type(exc).__name__},
        )

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(max(1, self.settings.max_retries)),
            wait=wait_exponential_jitter(initial=2.0, max=60.0),
            retry=retry_if_exception_type(_RETRYABLE),
            before_sleep=self._log_retry,
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                async with self._sem:
                    t0 = time.perf_counter()
                    resp = await self._client.chat.completions.create(
                        model=self.cfg.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=0,
                        max_tokens=self.settings.max_output_tokens,
                    )
                    latency = time.perf_counter() - t0
        return self._parse(resp, messages, latency)

    @staticmethod
    def _parse(resp: Any, messages: list[dict[str, str]], latency: float) -> LLMResponse:
        """Tolerant of malformed responses: missing choices/content/usage never raise."""
        text = ""
        finish: str | None = None
        choices = getattr(resp, "choices", None) or []
        if choices:
            first = choices[0]
            finish = getattr(first, "finish_reason", None)
            message = getattr(first, "message", None)
            content = getattr(message, "content", None)
            text = content if isinstance(content, str) else ""

        usage = getattr(resp, "usage", None)
        prompt = _int(getattr(usage, "prompt_tokens", None))
        completion = _int(getattr(usage, "completion_tokens", None))
        cached = _int(getattr(usage, "prompt_cache_hit_tokens", None))  # DeepSeek
        if cached is None:
            cached = _int(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None))  # OpenAI-style

        estimated = prompt is None or completion is None
        if prompt is None:
            prompt = count_tokens("\n".join(m["content"] for m in messages))
        if completion is None:
            completion = count_tokens(text)
        return LLMResponse(text, prompt, completion, cached or 0, estimated, latency, finish)


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


@dataclass
class RunStats:
    issues_total: int = 0
    pairs_skipped: int = 0
    pairs_ok: int = 0
    pairs_error: int = 0
    errors: list[str] = field(default_factory=list)


class BenchmarkRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        store: ResultStore,
        pruner: ContextPruner,
        repo_path: Path,
        pipelines: list[str],
        concurrency: int,
        b_layout: Layout = "naive",
        on_record: Callable[[RunRecord], None] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.pruner = pruner
        self.repo_path = repo_path.resolve()
        self.concurrency = max(1, concurrency)
        self.b_layout: Layout = b_layout
        self.on_record = on_record
        self.worker_cfg = settings.worker_config()
        self.frontier_cfg = settings.frontier_config() if settings.pipeline_a_mode == "live" and PIPELINE_A in pipelines else None

        names = list(dict.fromkeys(pipelines))
        if PIPELINE_A in names and settings.pipeline_a_mode == "simulate" and PIPELINE_B not in names:
            log.warning("Pipeline A (simulate) re-prices Pipeline B's response; adding B to the run")
            names.append(PIPELINE_B)
        self.pipelines = names

        self.worker = LLMClient(self.worker_cfg, settings, self.concurrency)
        self.frontier = LLMClient(self.frontier_cfg, settings, self.concurrency) if self.frontier_cfg else None

    # ------------------------------------------------------------------ public
    async def run(self, issues: list[Issue]) -> RunStats:
        stats = RunStats(issues_total=len(issues))
        completed = await asyncio.to_thread(self.store.completed_pairs)

        work: list[tuple[Issue, set[str]]] = []
        for issue in issues:
            todo = {p for p in self.pipelines if (issue.instance_id, p) not in completed}
            if todo:
                work.append((issue, todo))
        stats.pairs_skipped = len(issues) * len(self.pipelines) - sum(len(t) for _i, t in work)
        log.info("starting run", extra={"issues": len(issues), "to_run": len(work), "skipped_pairs": stats.pairs_skipped})

        gate = asyncio.Semaphore(self.concurrency)

        async def guarded(issue: Issue, todo: set[str]) -> list[RunRecord]:
            async with gate:
                try:
                    return await self._process_issue(issue, todo)
                except Exception as exc:  # last line of defence: one issue must never kill the batch
                    log.exception("unhandled error processing issue", extra={"instance_id": issue.instance_id})
                    return [await self._persist(self._error_record(issue.instance_id, p, f"unhandled: {exc!r}")) for p in todo]

        try:
            for batch in asyncio.as_completed([guarded(i, t) for i, t in work]):
                for rec in await batch:
                    if rec.status == "ok":
                        stats.pairs_ok += 1
                    else:
                        stats.pairs_error += 1
                        stats.errors.append(f"{rec.instance_id}/{rec.pipeline_name}: {rec.error}")
        finally:
            await self.worker.aclose()
            if self.frontier:
                await self.frontier.aclose()
        return stats

    # ------------------------------------------------------------------ internals
    async def _persist(self, rec: RunRecord) -> RunRecord:
        await asyncio.to_thread(self.store.save, rec)
        return rec

    def _error_record(self, instance_id: str, pipeline: str, error: str) -> RunRecord:
        return RunRecord(instance_id=instance_id, pipeline_name=pipeline, status="error", error=error[:1000])

    async def _process_issue(self, issue: Issue, todo: set[str]) -> list[RunRecord]:
        try:
            bundle = await asyncio.to_thread(self.pruner.build, issue.instance_id, issue.problem_statement)
        except Exception as exc:
            log.exception("context build failed", extra={"instance_id": issue.instance_id})
            return [await self._persist(self._error_record(issue.instance_id, p, f"context build failed: {exc!r}")) for p in todo]

        if not bundle.files:
            return [await self._persist(self._error_record(issue.instance_id, p, "retrieval found no candidate files")) for p in todo]

        gold = diff_files(issue.gold_patch) or None
        gold_in_ctx = None if gold is None else bool(gold & set(bundle.retrieved_paths))
        ctx = _Ctx(issue, bundle, gold, gold_in_ctx)

        simulate_a = self.settings.pipeline_a_mode == "simulate"
        need_b = PIPELINE_B in todo or (PIPELINE_A in todo and simulate_a)
        tasks: dict[str, asyncio.Task[RunRecord]] = {}

        if need_b:
            tasks[PIPELINE_B] = asyncio.create_task(self._pipeline_b(ctx, persist=PIPELINE_B in todo))
        if PIPELINE_C in todo:
            tasks[PIPELINE_C] = asyncio.create_task(self._pipeline_c(ctx))
        if PIPELINE_A in todo:
            tasks[PIPELINE_A] = asyncio.create_task(self._pipeline_a(ctx, tasks.get(PIPELINE_B)))

        await asyncio.gather(*tasks.values())
        return [tasks[p].result() for p in todo if p in tasks]

    def _announce(self, rec: RunRecord) -> RunRecord:
        log.info(
            "pipeline finished",
            extra={
                "instance_id": rec.instance_id, "pipeline": rec.pipeline_name, "status": rec.status,
                "input_tokens": rec.input_tokens, "output_tokens": rec.output_tokens,
                "cost_usd": round(rec.cost_usd, 6), "valid_git_diff": rec.valid_git_diff,
                "error": rec.error,
            },
        )
        if self.on_record:
            try:
                self.on_record(rec)
            except Exception:
                log.exception("on_record callback failed")
        return rec

    async def _call(
        self,
        ctx: "_Ctx",
        pipeline: str,
        client: LLMClient,
        *,
        context: str,
        context_tokens: int,
        layout: Layout,
        extra_latency: float = 0.0,
        saved_pct: float = 0.0,
        persist: bool = True,
    ) -> RunRecord:
        try:
            messages = build_messages(context, ctx.issue.problem_statement, layout)
            resp = await client.complete(messages)
            patch = extract_patch(resp.text)
            valid, apply_err = await git_apply_check(self.repo_path, patch, recount=self.settings.git_apply_recount)
            touched = diff_files(patch)
            cfg = client.cfg
            usd = price_usd(cfg, resp.prompt_tokens, resp.cached_tokens, resp.completion_tokens)
            err_note = None
            if not patch:
                err_note = f"model returned no diff (finish_reason={resp.finish_reason})"
            rec = RunRecord(
                instance_id=ctx.issue.instance_id,
                pipeline_name=pipeline,
                status="ok",
                error=err_note,
                model=cfg.model,
                input_tokens=resp.prompt_tokens,
                cached_input_tokens=resp.cached_tokens,
                output_tokens=resp.completion_tokens,
                context_tokens_raw=ctx.bundle.raw_tokens,
                context_tokens_pruned=context_tokens,
                tokens_saved_pct=saved_pct,
                cost_usd=usd,
                cost_inr=usd * self.settings.usd_to_inr,
                latency_sec=resp.latency_sec + extra_latency,
                valid_git_diff=valid,
                git_apply_error=apply_err,
                gold_file_match=None if ctx.gold is None else bool(touched & ctx.gold),
                gold_in_context=ctx.gold_in_ctx,
                usage_estimated=resp.usage_estimated,
                raw_truncated=ctx.bundle.raw_truncated,
                retrieved_files=",".join(ctx.bundle.retrieved_paths),
                raw_patch=patch,
            )
        except Exception as exc:  # malformed responses, auth errors, context-length errors, exhausted retries...
            log.warning(
                "pipeline call failed",
                extra={"instance_id": ctx.issue.instance_id, "pipeline": pipeline, "error": f"{type(exc).__name__}: {exc}"},
            )
            rec = self._error_record(ctx.issue.instance_id, pipeline, f"{type(exc).__name__}: {exc}")
            rec.retrieved_files = ",".join(ctx.bundle.retrieved_paths)
        if persist:
            await self._persist(rec)
            self._announce(rec)
        return rec

    async def _pipeline_b(self, ctx: "_Ctx", *, persist: bool) -> RunRecord:
        if not persist:  # B already checkpointed; reuse its stored response for Pipeline A's re-pricing
            existing = await asyncio.to_thread(self.store.get_record, ctx.issue.instance_id, PIPELINE_B)
            if existing is not None and existing.status == "ok":
                return existing
        return await self._call(
            ctx, PIPELINE_B, self.worker,
            context=ctx.bundle.raw_context, context_tokens=ctx.bundle.raw_tokens,
            layout=self.b_layout, persist=persist,
        )

    async def _pipeline_c(self, ctx: "_Ctx") -> RunRecord:
        return await self._call(
            ctx, PIPELINE_C, self.worker,
            context=ctx.bundle.pruned_context, context_tokens=ctx.bundle.pruned_tokens,
            layout="cached", extra_latency=ctx.bundle.prune_sec, saved_pct=ctx.bundle.reduction_pct,
        )

    async def _pipeline_a(self, ctx: "_Ctx", b_task: "asyncio.Task[RunRecord] | None") -> RunRecord:
        if self.frontier is not None:  # live mode
            return await self._call(
                ctx, PIPELINE_A, self.frontier,
                context=ctx.bundle.raw_context, context_tokens=ctx.bundle.raw_tokens, layout="naive",
            )

        # simulate mode: same prompt, same tokens as B; only the price list changes.
        assert b_task is not None
        b = await b_task
        if b.status != "ok":
            rec = self._error_record(ctx.issue.instance_id, PIPELINE_A, f"depends on Pipeline B which failed: {b.error}")
        else:
            s = self.settings
            usd = (b.input_tokens * s.frontier_price_input + b.output_tokens * s.frontier_price_output) / 1_000_000
            rec = b.model_copy(
                update={
                    "pipeline_name": PIPELINE_A,
                    "model": f"frontier-simulated (tokens from {b.model})",
                    "cached_input_tokens": 0,  # naive baseline: no caching discount
                    "tokens_saved_pct": 0.0,
                    "cost_usd": usd,
                    "cost_inr": usd * s.usd_to_inr,
                }
            )
        await self._persist(rec)
        return self._announce(rec)


@dataclass(slots=True)
class _Ctx:
    issue: Issue
    bundle: ContextBundle
    gold: set[str] | None
    gold_in_ctx: bool | None
