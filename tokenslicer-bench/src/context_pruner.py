"""Token-compression engine: issue text -> top-K files (BM25) -> AST-pruned context.

Pipeline
    A. Extract signals (paths, modules, classes, functions, stack-trace frames) from the issue
       using regex plus Python's `ast` on any code that appears in the issue.
    B. Rank every candidate .py file in the repo with BM25 (tests/docs excluded unless the issue
       mentions them) and keep the top-K.
    C. Parse each selected file with `ast`; keep imports, class signatures, sibling method signatures
       and the top relevant functions (+/- N lines of surrounding context); elide the rest.

Everything is deterministic: same repo + same issue -> byte-identical output.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import logging
import os
import re
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rank_bm25 import BM25Okapi

from .models import ContextBundle, FileContext, IssueSignals
from .tokens import count_tokens, tokenizer_name

log = logging.getLogger("tokenslicer.pruner")

# --------------------------------------------------------------------------------------
# Repo walking
# --------------------------------------------------------------------------------------

_ALWAYS_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", "build", "dist",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", ".eggs", "static",
}
_TEST_DIRS = {"tests", "test", "testing", "helm_tests", "kubernetes_tests", "system_tests", "docker_tests"}
_DOC_DIRS = {"docs", "doc", "docs_src", "dev", "example_dags", "examples"}
_MAX_FILE_BYTES = 1_000_000  # skip generated/vendored monsters
_MAX_INDEX_CHARS = 60_000  # per-file cap on text fed to BM25 (memory guard for 10k+ file repos)


def _classify(rel_parts: tuple[str, ...]) -> tuple[bool, bool]:
    """Return (is_test, is_doc) for a repo-relative path."""
    name = rel_parts[-1]
    dirs = {p.lower() for p in rel_parts[:-1]}
    is_test = (
        bool(dirs & _TEST_DIRS)
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )
    is_doc = bool(dirs & _DOC_DIRS)
    return is_test, is_doc


# --------------------------------------------------------------------------------------
# Tokenisation helpers
# --------------------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = set(keyword.kwlist) | set(dir(builtins)) | {
    "self", "cls", "args", "kwargs", "none", "true", "false", "str", "int", "dict", "list", "set",
    "the", "and", "for", "with", "this", "that", "from", "import", "def", "class", "return",
    "when", "not", "are", "was", "has", "have", "but", "can", "does", "should", "would", "run",
    "get", "set", "use", "using", "used", "issue", "error", "bug", "expected", "actual", "version",
    "python", "code", "file", "line", "call", "name", "value", "data", "test", "example", "see",
}


def _split_ident(ident: str) -> list[str]:
    parts: list[str] = []
    for chunk in ident.split("_"):
        if chunk:
            parts.extend(m.group(0).lower() for m in _CAMEL_SPLIT.finditer(chunk))
    return parts


def tokenize_text(text: str) -> list[str]:
    """Lower-cased identifiers plus their snake/camel sub-words. Interned to keep memory flat."""
    out: list[str] = []
    for m in _IDENT.finditer(text):
        ident = m.group(0)
        low = ident.lower()
        if low not in _STOP:
            out.append(sys.intern(low))
        parts = _split_ident(ident)
        if len(parts) > 1:
            out.extend(sys.intern(p) for p in parts if len(p) > 2 and p not in _STOP)
    return out


def _path_tokens(rel_posix: str) -> list[str]:
    out: list[str] = []
    for part in rel_posix.replace(".py", "").split("/"):
        out.extend(sys.intern(t) for t in tokenize_text(part))
    return out


# --------------------------------------------------------------------------------------
# Step A: signal extraction
# --------------------------------------------------------------------------------------

_PATH_RE = re.compile(r"(?<![\w.])((?:[\w.\-]+/)*[\w\-]+\.py)\b")
_TRACE_RE = re.compile(r'File "([^"]+\.py)", line (\d+), in ([A-Za-z_]\w*)')
_DOTTED_RE = re.compile(r"\b((?:[A-Za-z_]\w*\.){1,}[A-Za-z_]\w*)\b")
_CLASS_RE = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+|[A-Z]{2,}[a-z]\w*)\b")
_EXC_RE = re.compile(r"\b([A-Z]\w*(?:Error|Exception|Warning|Exit))\b")
_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*\(")
_BACKTICK_RE = re.compile(r"`([A-Za-z_][\w.]*)`")
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)
_TESTS_MENTION = re.compile(r"(?:\bpytest\b|\bunittest\b|\btests?/|\btest_\w+|\bconftest\b)", re.I)
_DOCS_MENTION = re.compile(r"(?:\bdocs?/|\.rst\b|\bsphinx\b)", re.I)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        seen.setdefault(it, None)
    return list(seen)


def _ast_names_from_code(code: str) -> tuple[list[str], list[str], list[str]]:
    """Parse code-ish text with `ast`; return (modules, classes, functions).

    Tries the whole block first; on SyntaxError falls back to line-by-line parsing so that
    traceback source lines (`    self.dag.run(x)`) and pasted one-liners still yield names.
    """
    modules: list[str] = []
    classes: list[str] = []
    funcs: list[str] = []

    def visit(tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.append(node.module)
                    for a in node.names:
                        (classes if a.name[:1].isupper() else funcs).append(a.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(node.name)
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name:
                    (classes if name[:1].isupper() else funcs).append(name)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                # e.g. DagRun.find -> keep both halves as candidates
                if node.value.id[:1].isupper():
                    classes.append(node.value.id)
                (classes if node.attr[:1].isupper() else funcs).append(node.attr)

    try:
        visit(ast.parse(code))
        return modules, classes, funcs
    except (SyntaxError, ValueError, RecursionError):
        pass

    for raw in code.splitlines()[:2000]:
        line = textwrap.dedent(raw).strip()
        if len(line) < 4 or len(line) > 400:
            continue
        try:
            visit(ast.parse(line))
        except (SyntaxError, ValueError, RecursionError):
            continue
    return modules, classes, funcs


def extract_signals(problem_statement: str) -> IssueSignals:
    text = problem_statement
    paths: list[str] = []
    functions: list[str] = []
    classes: list[str] = []
    keywords: list[str] = []
    modules: list[str] = []

    for path, _line, func in _TRACE_RE.findall(text):
        paths.append(path)
        if func != "<module>" and func.lower() not in _STOP:
            functions.append(func)

    paths.extend(m.group(1) for m in _PATH_RE.finditer(text))
    keywords.extend(_EXC_RE.findall(text))
    classes.extend(_CLASS_RE.findall(text))
    functions.extend(_CALL_RE.findall(text))

    for m in _BACKTICK_RE.finditer(text):
        token = m.group(1)
        if "." in token:
            modules.append(token)
        elif token[:1].isupper():
            classes.append(token)
        elif "_" in token:
            functions.append(token)

    modules.extend(_DOTTED_RE.findall(text))

    # Python `ast` over fenced code blocks, and over the whole text line-by-line (tracebacks).
    code_blobs = _FENCE_RE.findall(text) or []
    code_blobs.append(text)
    for blob in code_blobs:
        mods, cls, fns = _ast_names_from_code(blob)
        modules.extend(mods)
        classes.extend(cls)
        functions.extend(fns)

    def clean(names: list[str]) -> list[str]:
        return _ordered_unique([n for n in names if len(n) > 2 and n.lower() not in _STOP])

    return IssueSignals(
        paths=_ordered_unique([p.replace("\\", "/") for p in paths]),
        modules=_ordered_unique(
            [m for m in modules if m.count(".") >= 1 and len(m) < 120 and not m.endswith((".py", ".md", ".rst", ".txt"))]
        ),
        classes=clean(classes),
        functions=clean(functions),
        keywords=clean(keywords),
        mentions_tests=bool(_TESTS_MENTION.search(text)),
        mentions_docs=bool(_DOCS_MENTION.search(text)),
    )


# --------------------------------------------------------------------------------------
# Step B: BM25 index over the repository
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class IndexedFile:
    rel: str  # posix path relative to repo root
    parts: tuple[str, ...]
    is_test: bool
    is_doc: bool


class RepoIndex:
    """BM25 index over every .py file. Built once per process, queried per issue."""

    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root.resolve()
        self.files: list[IndexedFile] = []
        self._by_basename: dict[str, list[int]] = {}
        started = time.perf_counter()
        corpus: list[list[str]] = []

        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames if d not in _ALWAYS_SKIP_DIRS and not d.startswith(".")
            )
            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                full = Path(dirpath) / fname
                try:
                    if full.stat().st_size > _MAX_FILE_BYTES:
                        continue
                    text = full.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                rel_parts = full.relative_to(self.root).parts
                rel = "/".join(rel_parts)
                is_test, is_doc = _classify(rel_parts)
                idx = len(self.files)
                self.files.append(IndexedFile(rel, rel_parts, is_test, is_doc))
                self._by_basename.setdefault(fname, []).append(idx)
                # Path tokens are repeated so a module name in the path outweighs incidental mentions.
                corpus.append(_path_tokens(rel) * 3 + tokenize_text(text[:_MAX_INDEX_CHARS]))

        if not self.files:
            raise FileNotFoundError(f"No .py files found under {self.root}")
        self.bm25 = BM25Okapi(corpus)
        log.info(
            "repo index built",
            extra={"files": len(self.files), "seconds": round(time.perf_counter() - started, 2)},
        )

    # -- path/module matching -------------------------------------------------------------
    def _match_path(self, mention: str) -> list[int]:
        """Indices of files whose path ends with `mention`, matching >=2 trailing components
        (or the whole mention when it is just a basename). Handles absolute stack-trace paths."""
        parts = [p for p in PurePosixPath(mention).parts if p not in ("/", ".", "..")]
        if not parts:
            return []
        hits: list[int] = []
        for idx in self._by_basename.get(parts[-1], []):
            f_parts = self.files[idx].parts
            common = 0
            for a, b in zip(reversed(parts), reversed(f_parts)):
                if a != b:
                    break
                common += 1
            if common >= min(2, len(parts)):
                hits.append(idx)
        return hits

    def _match_module(self, dotted: str) -> list[int]:
        parts = dotted.split(".")
        for k in range(len(parts), 1, -1):
            as_path = "/".join(parts[:k])
            hits = self._match_path(as_path + ".py") + self._match_path(as_path + "/__init__.py")
            if hits:
                return hits
        return []

    # -- ranking --------------------------------------------------------------------------
    def rank(self, signals: IssueSignals, problem_statement: str, top_k: int) -> list[tuple[int, float]]:
        query: list[str] = []
        for p in signals.paths:
            query.extend(_path_tokens(p) * 3)
        for name in signals.classes:
            query.extend(tokenize_text(name) * 3)
        for name in signals.functions:
            query.extend(tokenize_text(name) * 2)
        for name in signals.keywords:
            query.extend(tokenize_text(name) * 2)
        for mod in signals.modules:
            query.extend(tokenize_text(mod.replace(".", " ")) * 2)
        query.extend(tokenize_text(problem_statement)[:400])
        if not query:
            return []

        scores = self.bm25.get_scores(query).astype(float)
        top = float(scores.max()) if len(scores) else 0.0
        bonus = top if top > 0 else 1.0

        # Explicitly named files/modules get a decisive boost.
        explicit: set[int] = set()
        for mention in signals.paths:
            explicit.update(self._match_path(mention))
        for dotted in signals.modules:
            explicit.update(self._match_module(dotted))
        for idx in explicit:
            scores[idx] += bonus

        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.files[i].rel))
        picked: list[tuple[int, float]] = []
        for idx in order:
            f = self.files[idx]
            if scores[idx] <= 0:
                break
            if f.is_test and not (signals.mentions_tests or idx in explicit):
                continue
            if f.is_doc and not (signals.mentions_docs or idx in explicit):
                continue
            picked.append((idx, float(scores[idx])))
            if len(picked) >= top_k:
                break
        return picked


_INDEX_CACHE: dict[Path, RepoIndex] = {}
_INDEX_LOCK = threading.Lock()


def get_index(repo_root: Path) -> RepoIndex:
    key = repo_root.resolve()
    with _INDEX_LOCK:
        if key not in _INDEX_CACHE:
            _INDEX_CACHE[key] = RepoIndex(key)
        return _INDEX_CACHE[key]


# --------------------------------------------------------------------------------------
# Step C: AST pruning
# --------------------------------------------------------------------------------------

_FuncNode = ast.FunctionDef | ast.AsyncFunctionDef


def _start_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", None) or []
    first = min([node.lineno] + [d.lineno for d in decorators])  # type: ignore[attr-defined]
    return first


def _header_end(node: ast.AST) -> int:
    """Last line of a class/def signature (the line before the first body statement)."""
    body = node.body  # type: ignore[attr-defined]
    first_body = body[0].lineno if body else node.lineno  # type: ignore[attr-defined]
    return max(node.lineno, first_body - 1)  # type: ignore[attr-defined]


def _collect_imports(body: list[ast.stmt], keep: set[int]) -> None:
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.If):
            _collect_imports(node.body, keep)
            _collect_imports(node.orelse, keep)
        elif isinstance(node, ast.Try):
            _collect_imports(node.body, keep)
            for handler in node.handlers:
                _collect_imports(handler.body, keep)
            _collect_imports(node.orelse, keep)
            _collect_imports(node.finalbody, keep)


def _function_score(node: _FuncNode, terms: set[str], func_names: set[str]) -> int:
    score = 0
    name_low = node.name.lower()
    if name_low in func_names:
        score += 6
    elif name_low in terms:
        score += 3
    score += sum(1 for t in _split_ident(node.name) if t in terms)

    seen: set[str] = set()
    for sub in ast.walk(node):
        ident: str | None = None
        if isinstance(sub, ast.Name):
            ident = sub.id
        elif isinstance(sub, ast.Attribute):
            ident = sub.attr
        elif isinstance(sub, ast.arg):
            ident = sub.arg
        if ident:
            low = ident.lower()
            if low in terms and low not in seen:
                seen.add(low)
                score += 1
    return score


def prune_source(
    source: str,
    signals: IssueSignals,
    *,
    context_lines: int = 15,
    max_functions: int = 5,
    sibling_signatures: bool = True,
) -> tuple[str, bool, list[str]]:
    """Return (pruned_source, parsed_ok, kept_function_names)."""
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        # Unparseable (py2, templated, etc.): keep the head of the file rather than nothing.
        head = lines[:200]
        tail_note = [f"# ... [lines 201-{len(lines)} pruned by TokenSlicer: file not parseable] ..."] if len(lines) > 200 else []
        return "\n".join(head + tail_note), False, []

    terms: set[str] = set()
    for group in (signals.classes, signals.functions, signals.keywords):
        for name in group:
            terms.add(name.lower())
            terms.update(_split_ident(name))
    for mod in signals.modules:
        for part in mod.split("."):
            terms.add(part.lower())
    terms -= _STOP
    func_names = {f.lower() for f in signals.functions}

    keep: set[int] = set()
    _collect_imports(tree.body, keep)

    # (score, lineno, node, enclosing class or None)
    candidates: list[tuple[int, int, _FuncNode, ast.ClassDef | None]] = []
    classes: list[ast.ClassDef] = []

    def walk(body: list[ast.stmt], owner: ast.ClassDef | None) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                classes.append(node)
                walk(node.body, node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates.append((_function_score(node, terms, func_names), node.lineno, node, owner))
            elif isinstance(node, (ast.If, ast.Try)):
                walk(node.body, owner)
                walk(getattr(node, "orelse", []), owner)
                for handler in getattr(node, "handlers", []):
                    walk(handler.body, owner)
                walk(getattr(node, "finalbody", []), owner)

    walk(tree.body, None)

    # Class signatures (decorators + `class X(Base):` header) for every class.
    for cls in classes:
        keep.update(range(_start_line(cls), _header_end(cls) + 1))

    ranked = sorted((c for c in candidates if c[0] > 0), key=lambda c: (-c[0], c[1]))[:max_functions]
    kept_names: list[str] = []
    relevant_classes: set[int] = set()
    for _score, _ln, fn, owner in ranked:
        lo = max(1, _start_line(fn) - context_lines)
        hi = min(len(lines), (fn.end_lineno or fn.lineno) + context_lines)
        keep.update(range(lo, hi + 1))
        kept_names.append(f"{owner.name}.{fn.name}" if owner else fn.name)
        if owner is not None:
            relevant_classes.add(id(owner))

    # Signatures (only) of the other callables, so the model still sees the surrounding API.
    # With relevant hits: signatures of methods in the same classes. With none: an API overview of the file.
    if sibling_signatures:
        for _score, _ln, fn, owner in candidates:
            in_relevant_class = owner is not None and id(owner) in relevant_classes
            if in_relevant_class or not ranked:
                keep.update(range(_start_line(fn), _header_end(fn) + 1))

    keep = {n for n in keep if 1 <= n <= len(lines)}
    if not keep:
        return "\n".join(lines[:60]), True, kept_names

    # Fill tiny gaps (<=2 lines) to avoid noisy elision markers.
    ordered = sorted(keep)
    out: list[str] = []
    prev = 0
    for n in ordered:
        gap = n - prev - 1
        if prev and gap > 0:
            if gap <= 2:
                out.extend(lines[prev:n - 1])
            else:
                out.append(f"# ... [lines {prev + 1}-{n - 1} pruned by TokenSlicer] ...")
        elif not prev and n > 1:
            out.append(f"# ... [lines 1-{n - 1} pruned by TokenSlicer] ...")
        out.append(lines[n - 1])
        prev = n
    if prev < len(lines):
        out.append(f"# ... [lines {prev + 1}-{len(lines)} pruned by TokenSlicer] ...")
    return "\n".join(out), True, kept_names


# --------------------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------------------


def _render_file(rel: str, body: str, note: str = "") -> str:
    suffix = f" ({note})" if note else ""
    return f"### FILE: {rel}{suffix}\n```python\n{body}\n```\n"


def _fit_to_budget(blocks: list[tuple[str, str]], cap_tokens: int) -> tuple[list[tuple[str, str]], bool]:
    """Water-filling truncation: small files stay whole, the biggest files absorb the cut."""
    sizes = [count_tokens(body) for _rel, body in blocks]
    if sum(sizes) <= cap_tokens:
        return blocks, False

    order = sorted(range(len(blocks)), key=lambda i: sizes[i])
    budget = cap_tokens
    alloc = [0] * len(blocks)
    remaining = len(blocks)
    for i in order:
        share = budget // max(remaining, 1)
        alloc[i] = min(sizes[i], share)
        budget -= alloc[i]
        remaining -= 1

    fitted: list[tuple[str, str]] = []
    for i, (rel, body) in enumerate(blocks):
        if alloc[i] >= sizes[i]:
            fitted.append((rel, body))
            continue
        lines = body.splitlines()
        ratio = alloc[i] / max(sizes[i], 1)
        keep_n = max(1, int(len(lines) * ratio))
        fitted.append(
            (rel, "\n".join(lines[:keep_n]) + f"\n# ... [lines {keep_n + 1}-{len(lines)} truncated: context budget] ...")
        )
    return fitted, True


class ContextPruner:
    def __init__(
        self,
        repo_path: Path,
        *,
        top_k: int = 5,
        context_lines: int = 15,
        max_functions_per_file: int = 5,
        sibling_signatures: bool = True,
        raw_token_cap: int = 110_000,
    ) -> None:
        self.repo_path = repo_path.resolve()
        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        self.top_k = top_k
        self.context_lines = context_lines
        self.max_functions_per_file = max_functions_per_file
        self.sibling_signatures = sibling_signatures
        self.raw_token_cap = raw_token_cap

    def warm(self) -> RepoIndex:
        """Build the BM25 index now (call once before concurrent use)."""
        return get_index(self.repo_path)

    def build(self, instance_id: str, problem_statement: str) -> ContextBundle:
        t0 = time.perf_counter()
        index = get_index(self.repo_path)
        signals = extract_signals(problem_statement)
        ranked = index.rank(signals, problem_statement, self.top_k)
        retrieval_sec = time.perf_counter() - t0

        raw_blocks: list[tuple[str, str]] = []
        sources: list[tuple[str, float, str]] = []
        for idx, score in ranked:
            rel = index.files[idx].rel
            try:
                text = (self.repo_path / rel).read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                log.warning("cannot read candidate file", extra={"file": rel, "error": str(exc)})
                continue
            raw_blocks.append((rel, text))
            sources.append((rel, score, text))

        raw_blocks, truncated = _fit_to_budget(raw_blocks, self.raw_token_cap)
        raw_context = "\n".join(_render_file(rel, body) for rel, body in raw_blocks)

        t1 = time.perf_counter()
        pruned_parts: list[str] = []
        files: list[FileContext] = []
        for (rel, score, text), (_rel2, raw_body) in zip(sources, raw_blocks):
            pruned, ok, kept = prune_source(
                text,
                signals,
                context_lines=self.context_lines,
                max_functions=self.max_functions_per_file,
                sibling_signatures=self.sibling_signatures,
            )
            pruned_parts.append(_render_file(rel, pruned, "pruned view; '# ... pruned' markers show elided line ranges"))
            files.append(
                FileContext(
                    path=rel,
                    bm25_score=round(score, 4),
                    raw_tokens=count_tokens(raw_body),
                    pruned_tokens=count_tokens(pruned),
                    parsed_ok=ok,
                    kept_functions=kept,
                )
            )
        prune_sec = time.perf_counter() - t1
        pruned_context = "\n".join(pruned_parts)

        raw_tokens = count_tokens(raw_context)
        pruned_tokens = count_tokens(pruned_context)
        reduction = round((1 - pruned_tokens / raw_tokens) * 100, 2) if raw_tokens else 0.0
        return ContextBundle(
            instance_id=instance_id,
            files=files,
            raw_context=raw_context,
            pruned_context=pruned_context,
            raw_tokens=raw_tokens,
            pruned_tokens=pruned_tokens,
            reduction_pct=reduction,
            raw_truncated=truncated,
            tokenizer=tokenizer_name(),
            retrieval_sec=round(retrieval_sec, 4),
            prune_sec=round(prune_sec, 4),
            signals=signals,
        )
