"""Shared pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

# Canonical pipeline names (these are the checkpoint keys stored in DuckDB).
PIPELINE_A = "A_naive_frontier"
PIPELINE_B = "B_raw_cheap_worker"
PIPELINE_C = "C_tokenslicer_cheap_worker"
PIPELINE_LABELS = {"A": PIPELINE_A, "B": PIPELINE_B, "C": PIPELINE_C}
PIPELINE_ORDER = [PIPELINE_A, PIPELINE_B, PIPELINE_C]


class Issue(BaseModel):
    instance_id: str
    repo: str = ""
    problem_statement: str
    base_commit: str | None = None
    gold_patch: str | None = None


class IssueSignals(BaseModel):
    """Everything we could pull out of the issue text (regex + `ast`)."""

    paths: list[str] = Field(default_factory=list)  # file paths, incl. stack-trace frames
    modules: list[str] = Field(default_factory=list)  # dotted module/attribute paths
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)  # exceptions, other identifiers
    mentions_tests: bool = False
    mentions_docs: bool = False


class FileContext(BaseModel):
    path: str
    bm25_score: float
    raw_tokens: int
    pruned_tokens: int
    parsed_ok: bool
    kept_functions: list[str] = Field(default_factory=list)


class ContextBundle(BaseModel):
    instance_id: str
    files: list[FileContext]
    raw_context: str
    pruned_context: str
    raw_tokens: int
    pruned_tokens: int
    reduction_pct: float
    raw_truncated: bool
    tokenizer: str
    retrieval_sec: float
    prune_sec: float
    signals: IssueSignals

    @property
    def retrieved_paths(self) -> list[str]:
        return [f.path for f in self.files]


class RunRecord(BaseModel):
    """One row of the `runs` table."""

    instance_id: str
    pipeline_name: str
    status: str  # "ok" | "error"
    error: str | None = None
    model: str | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    context_tokens_raw: int = 0
    context_tokens_pruned: int = 0
    tokens_saved_pct: float = 0.0
    cost_usd: float = 0.0
    cost_inr: float = 0.0
    latency_sec: float = 0.0
    valid_git_diff: bool = False
    git_apply_error: str | None = None
    gold_file_match: bool | None = None  # patch touches >=1 gold-patched file (None when no gold patch)
    gold_in_context: bool | None = None  # retrieval put >=1 gold file in the context
    usage_estimated: bool = False  # True when the API returned no usage block
    raw_truncated: bool = False
    retrieved_files: str = ""
    raw_patch: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
