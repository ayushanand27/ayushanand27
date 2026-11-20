# TokenSlicer-Bench

Open-source benchmark that measures how much an enterprise can cut LLM bug-fixing bills on a huge repo
(Apache Airflow, ~8.7k Python files) by combining two levers, and what each lever is worth on its own:

| Lever | What it does | Where it shows up |
|---|---|---|
| **Model arbitrage** | Send the work to a cheap open-weight model (DeepSeek, Groq, OpenRouter) instead of a frontier model | A → B |
| **Context pruning** | BM25 retrieval + AST pruning, so the model sees ~⅓ of the tokens | B → C |

## Read this first: three honest caveats

1. **SWE-bench_Lite contains no Airflow issues.** Its 12 repos are astropy, django, matplotlib, seaborn, flask, requests,
   xarray, pylint, pytest, scikit-learn, sphinx, sympy. Filtering it for `airflow` returns nothing (the loader
   tells you so and lists the repos). Use `scripts/mine_airflow_issues.py` to build real Airflow
   instances from GitHub, or benchmark a repo that *is* in SWE-bench Lite (e.g. `--repo-filter flask`).
2. **"Valid patch" ≠ "correct patch".** The tool checks `git apply --check` and whether the patch touches a file the human fix
   touched. It does **not** run tests, so it cannot prove "no loss of accuracy". It *can* show whether pruning made the
   patches less well-formed or less well-localised than raw context (compare B vs C). For a resolve-rate claim, feed the saved
   patches (`raw_patch` column) to the official SWE-bench harness.
3. **Savings depend on your numbers.** Prices in `.env` are defaults, not quotes. Verify them. Context pruning alone gave
   ~58–67% fewer context tokens on four hand-written Airflow-style issues (default settings; see "Tuning"); the 90%+ figures appear
   when model arbitrage is stacked on top. Your dataset will give different numbers. The scorecard reports what was measured.

## Architecture

```
issue text ──► signals (regex + ast) ──► BM25 over all .py files ──► top-5 files
                                                                        │
                                   ┌────────────────────────────────────┤
                                   ▼                                    ▼
                          RAW full files                      AST-PRUNED view
                    (A and B send this)                 (C sends this)
                                   │                                    │
        A: frontier price list ◄───┤                                    │
           (simulate: re-price B)  ▼                                    ▼
        B: cheap worker, naive prompt layout            C: cheap worker, cache-optimised layout
                                   └────────────────┬───────────────────┘
                                                    ▼
                        extract diff ─► `git apply --check` ─► SQLite ledger ─► DuckDB ─► scorecard
```

| File | Role |
|---|---|
| `src/context_pruner.py` | Signal extraction, BM25 index (built once, ~14 s / 172 MB on Airflow), AST pruning |
| `src/router_engine.py` | Async OpenAI-compatible client (`openai` + `httpx`), `tenacity` backoff, prompts, patch extraction/validation, pipelines A/B/C |
| `src/storage.py` | SQLite WAL checkpoint ledger + DuckDB analytics store, resume logic |
| `src/report.py` | `rich` table, `results/scorecard.md`, `results/summary.json` |
| `src/dataset.py` | Hugging Face or JSONL issue loading |
| `scripts/mine_airflow_issues.py` | Build real Airflow issue/fix pairs from GitHub (needs network + optional `GITHUB_TOKEN`) |
| `run_benchmark.py` | CLI |

### How pruning works

1. **Signals**: file paths and stack-trace frames (`File "...", line N, in func`), CamelCase classes, `name(` calls, backticked identifiers, dotted
   module paths, exception names, and names found by running `ast.parse` on code in the issue (whole block first, then line by line so
   traceback source lines still parse).
2. **Retrieval**: BM25 (`rank_bm25`) over identifiers + sub-words of every `.py` file, path tokens up-weighted. Files named by the issue
   (suffix-matched, so absolute stack-trace paths work) get a decisive boost. Tests, docs and `example_dags` are excluded unless the issue
   mentions them. Top 5 are kept.
3. **Pruning** per file: keep all imports, every class signature, the one-line signatures of sibling methods in relevant classes, and the top-5
   functions scored by name/identifier overlap with the signals, each with ±15 lines of context. Everything else becomes
   `# ... [lines a-b pruned by TokenSlicer] ...`. Unparseable files fall back to their first 200 lines.

### The 3-tier cost comparison (same issues, same retrieved files)

| Pipeline | Context | Prompt layout | Pricing | Notes |
|---|---|---|---|---|
| **A Naive Frontier** | raw files | naive | `FRONTIER_PRICE_INPUT/OUTPUT` (default $3 / $15 per 1M) | `PIPELINE_A_MODE=simulate` (default): re-prices **B's real token counts**; zero frontier spend, but A's patch, latency and validity are B's. `live`: really calls `FRONTIER_*`. |
| **B Raw + Cheap Worker** | raw files | naive: issue first, instructions last, so nothing is shared across issues | real usage × worker price list | Isolates model arbitrage. |
| **C TokenSlicer + Cheap Worker** | AST-pruned | cached: static system prompt (role + rules + worked example) first, per-issue payload last | real usage, cached-prefix tokens at the cache-hit price | Isolates pruning + cache-friendly layout. C's latency includes AST-pruning time. |

`--b-layout cached` gives B the same cache-friendly layout as C so that B → C measures **pruning alone**. Cache discounts are only
credited when the provider reports them (`prompt_cache_hit_tokens` for DeepSeek, `prompt_tokens_details.cached_tokens` for
OpenAI-style APIs). Raw context is truncated to `RAW_CONTEXT_TOKEN_CAP` (default 110k) so it fits the worker's window; truncated rows are flagged in the report.

## Setup

```bash
# 1. Get the code and install (Python 3.11+)
cd tokenslicer-bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Shallow-clone the target repo (it must be its own git repo: `git apply --check` runs inside it)
git clone --depth 1 https://github.com/apache/airflow.git airflow

# 3. Configure credentials and prices
cp .env.example .env        # then edit: set DEEPSEEK_API_KEY (or OPENROUTER_/GROQ_), check prices, USD_TO_INR

# 4. Build a real Airflow issue set (SWE-bench_Lite has none)
export GITHUB_TOKEN=ghp_xxx   # optional but recommended (5000 req/h vs 60)
python scripts/mine_airflow_issues.py --out data/airflow_issues.jsonl --count 60
```

## Run

```bash
# Free check, no API key needed: retrieval + compression numbers only
python run_benchmark.py --repo-path ./airflow --issues-file data/airflow_issues.jsonl --limit 25 --dry-run

# The benchmark (resumable; Ctrl-C is safe)
python run_benchmark.py --repo-path ./airflow --issues-file data/airflow_issues.jsonl --limit 25 --concurrency 5

# Re-render the scorecard from the DB without calling any API
python run_benchmark.py --report-only

# Use a Hugging Face dataset instead (needs a repo that exists in it)
python run_benchmark.py --repo-path ./flask --repo-filter flask --limit 3 --concurrency 3
```

Other flags: `--pipelines A,B,C`, `--top-k 5`, `--context-lines 15`, `--max-functions 5`, `--no-sibling-signatures`, `--b-layout naive|cached`,
`--usd-inr 88`, `--results-dir results`, `-v`.

**Switch provider** in `.env`: `WORKER_PROVIDER=openrouter` (+ `OPENROUTER_API_KEY`) or `groq`, or `custom` with `WORKER_BASE_URL`, `WORKER_MODEL`,
`WORKER_API_KEY` and all three `WORKER_PRICE_*` values.

## Crash-proofing and resume

Every result is committed to a SQLite WAL ledger (`results/checkpoint.sqlite`, `synchronous=FULL`) **before** it is written to DuckDB
(`results/benchmark.duckdb`). On start the ledger is reconciled into DuckDB, so even a deleted DuckDB file is rebuilt. The key is
`(instance_id, pipeline_name)`; only `status='ok'` rows count as done, so API failures are retried on the next run and never re-bill finished work.
A malformed API response (no `choices`, `content: null`, missing `usage`) is recorded rather than raised; missing usage is estimated locally and flagged `usage_estimated`.

## Outputs

* `results/benchmark.duckdb`: table `runs` with `instance_id, pipeline_name, input_tokens, output_tokens, tokens_saved_pct, cost_usd, cost_inr, latency_sec, valid_git_diff, raw_patch` plus `cached_input_tokens, status, error, git_apply_error, gold_file_match, gold_in_context, …`
* `results/scorecard.md`, `results/summary.json`: total cost (USD/INR), avg latency, context token reduction, billed-input-token reduction, valid-patch rate, gold-file match. Computed on the **paired** set (issues finished by every pipeline).
* `results/benchmark.log.jsonl`: structured logs.

## Tuning compression

Measured on four hand-written Airflow-style issues against a fresh `apache/airflow` clone (tiktoken `cl100k_base`; illustrative, not a benchmark result):

| Settings | Context token reduction |
|---|---|
| default (`--max-functions 5`, ±15 lines, sibling signatures) | 65.9% |
| `--no-sibling-signatures` | 73.7% |
| `--max-functions 3 --context-lines 8 --no-sibling-signatures` | 80.6% |

More compression costs the model context; use B vs C valid-patch and gold-file rates to decide where your quality floor is.

## Limitations

* `git apply --check` runs against whatever the clone's working tree is. SWE-style historical issues were written against an older `base_commit`; a
  shallow clone at HEAD may already contain the fix or differ from what the issue describes. The benchmark is internally consistent (the model sees and patches the same tree), but it is not a historical replay.
* `GIT_APPLY_RECOUNT=false` (default) is strict: LLM-miscounted hunk headers fail. Set `true` to forgive them.
* Token counts for A are the worker tokenizer's, re-priced; a frontier model's tokenizer would differ somewhat.
* Provider caching is best-effort; concurrent first requests may miss, and some providers have no prefix caching.
