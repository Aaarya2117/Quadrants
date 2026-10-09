# Model Gate: Architecture

## 1. Design Principles

1. **Models are untrusted, the harness is deterministic.** All decisions come from graders, statistics, and policy code. No model judges another model in the MVP.
2. **Every decision is replayable.** Raw responses and configuration hashes are stored, so the decision can be recomputed without model calls.
3. **Separate what failed from why.** Free and constrained decoding are kept as separate outcomes so failure classes stay meaningful.
4. **Spend compute only when evidence is unclear.** Sequential statistics stop early on conclusive results.
5. **Swap-friendly edges.** Adapters, graders, and report renderers are plugins behind small interfaces.

## 2. System Context

```
 +-----------------+        +-----------------------------+
 | Engineer / PR   | -----> | Model Gate CLI / Action     |
 +-----------------+        +-----------------------------+
                                  |             |
                 +----------------+             +------------------+
                 v                                               v
     +-----------------------+                     +-----------------------+
     | Model endpoints       |                     | Artifact store        |
     | Ollama, vLLM, llama   |                     | (runs/ dir or CI      |
     | OpenAI-compatible API |                     |  artifact upload)     |
     +-----------------------+                     +-----------------------+
```

## 3. Component Overview

```
model_gate/
├── cli.py                  # Typer entry point: run, replay, report, init
├── config/
│   ├── loader.py           # YAML loading and validation (pydantic v2)
│   └── models.py           # Typed config and suite models
├── suite/
│   ├── schema.py           # Case definitions per kind
│   ├── importer.py         # JSONL trace import and sanitization
│   └── redact.py           # PII pattern scanning
├── adapters/
│   ├── base.py             # Adapter protocol: generate(), capabilities()
│   ├── ollama.py
│   └── openai_compat.py
├── runner/
│   ├── executor.py         # Async execution, concurrency limits, retries
│   ├── cache.py            # SQLite cache keyed by request hash
│   └── planner.py          # Expands cases x models x seeds x decoding modes
├── graders/
│   ├── base.py
│   ├── tool_call.py
│   ├── json_extract.py
│   ├── multimodal.py
│   └── taxonomy.py         # Failure class assignment
├── stats/
│   ├── paired.py           # McNemar, paired bootstrap
│   ├── sequential.py       # Alpha-spending, decision states
│   └── pareto.py           # Dominance and frontier computation
├── policy/
│   ├── engine.py           # Evaluates gate.yaml rules
│   └── decision.py         # pass | block | inconclusive, with reasons
├── hints/
│   ├── templates.py        # Fix-hint templates per failure class
│   └── verify.py           # Re-run affected cases, test shrinkage
├── report/
│   ├── markdown.py         # PR comment renderer
│   ├── html.py             # Static report with Pareto chart
│   └── json_out.py
├── bundle/
│   ├── writer.py           # Reproducibility bundle
│   └── replayer.py         # Recompute decision from bundle
└── ci/
    └── action/             # Composite GitHub Action wrapper
```

## 4. Core Data Model

```
Suite ──< Case ──< Expansion (case, model, seed, decoding)
                       │
                       └──> Response (raw output, usage, latency, cost)
                                 │
                                 └──> Outcome (pass/fail, failure_class, grader_version)

Outcome pairs (baseline, candidate) per case ──> Stats results ──> Policy decision
```

Key identifiers:

- `case_hash`: hash of the case definition
- `prompt_hash`: hash of the fully rendered request
- `expansion_key`: `(model_id, adapter_version, prompt_hash, seed, decoding_mode)`
- `bundle_id`: hash of the suite, policy, model configs, grader versions, and all `expansion_key`s

## 5. Key Interfaces

### Adapter

```python
class Adapter(Protocol):
    id: str                       # e.g. "ollama:gemma3:4b-it-q4_K_M"
    version: str

    def capabilities(self) -> Capabilities:
        """Declares: constrained_json, constrained_grammar, images, max_tokens."""

    async def generate(self, req: Request) -> Response:
        """Returns text, tool_calls, usage, latency_ms, raw payload."""
```

### Grader

```python
class Grader(Protocol):
    name: str
    version: str

    def grade(self, case: Case, response: Response) -> Outcome:
        """Deterministic. Returns passed, failure_class, and details."""
```

### Policy rule

```python
class Rule(BaseModel):
    metric: str
    max_drop: float | None = None
    max_increase_pct: float | None = None
    confidence: float = 0.95
    action: Literal["block", "warn"]
```

## 6. Statistical Design

- **Unit of analysis:** the paired case (baseline outcome, candidate outcome). Each case contributes K seed-level outcomes, aggregated to a case-level success rate per model.
- **Binary metrics** (tool-call accuracy, JSON validity): paired differences tested with McNemar's test on discordant pairs; confidence intervals via paired bootstrap over cases.
- **Continuous metrics** (latency, cost): paired bootstrap of mean or percentile difference.
- **Sequential monitoring:** after each batch of cases, update an always-valid bound on the difference. Decision states:
  - `block` when the lower bound on the drop exceeds `max_drop`
  - `pass` when the upper bound on the drop is below `max_drop` (non-inferiority)
  - `continue` otherwise, until the case budget is spent
  - `inconclusive` when the budget ends with neither bound resolved
- **Multiple rules:** alpha is split across rules with a Bonferroni-style allocation in the MVP.

## 7. Constrained-Decoding Probe

For each structured case, the planner adds a second expansion with `decoding_mode = constrained` if the adapter reports `constrained_json` or `constrained_grammar`. The schema from the case becomes the grammar or JSON schema. The diagnosis matrix is computed by joining free and constrained outcomes on the case key.

## 8. Concurrency and Caching

- Executor uses `asyncio` with a semaphore per provider and a token-bucket rate limit.
- Retries apply to transport errors only, with exponential backoff. Model-level errors are recorded as outcomes, not retried.
- Cache lookups happen before any network call. A cache hit records the original latency and marks `cached: true`, so timing reports stay honest.

## 9. Security and Privacy

- Secrets are read from environment variables. The config loader rejects inline keys.
- The sanitizer runs during import and before the bundle is written. It hashes or masks matched patterns and records only counts.
- Bundles store responses, which may contain echoed input. The README warns teams to use sanitized traces.
- The GitHub Action uses minimal permissions (`pull-requests: write`, `statuses: write`) and does not expose secrets to forked pull requests.

## 10. Extensibility

| Extension point | How to extend |
|-----------------|---------------|
| New backend | Implement `Adapter` |
| New task kind | Add a schema and a grader, register in `suite/schema.py` and `graders/` |
| New metric | Add an extractor in `stats/` and reference it in policy |
| New report | Implement a renderer in `report/` |
| New failure class | Add to `taxonomy.py` and a fix-hint template |

## 11. Deployment Modes

- **Local:** `model-gate run` on a developer machine with Ollama.
- **CI, recorded:** the action replays stored responses from `runs/recorded/`; cheap and deterministic.
- **CI, live:** the action calls the endpoint on a schedule or on demand, and writes a fresh bundle as an artifact.

## 12. Technology Choices

| Concern | Choice | Reason |
|---------|--------|--------|
| Language | Python 3.11+ | Stats and ML ecosystem; easy contributor onboarding |
| CLI | Typer | Typed commands with minimal boilerplate |
| Config and models | Pydantic v2 | Validation with clear error messages |
| HTTP | httpx (async) | Connection pooling and async support |
| Cache and storage | SQLite | Zero-setup, file-based, portable with the bundle |
| Statistics | NumPy, SciPy | Bootstrap and McNemar with standard implementations |
| Charts | Plotly (static export) or Vega-Lite spec | Self-contained HTML report |
| CI | GitHub composite Action | Works on any repository without a hosted service |

## 13. Known Architectural Risks

- **Constrained decoding differs across backends.** The diagnosis matrix is labeled with the backend, and results are not compared across backends unless flagged.
- **Sequential bounds are conservative.** Small suites will return `inconclusive` more often; the README documents suite-size guidance.
- **Recorded responses can drift from live behavior.** The bundle records the adapter version and the endpoint's model digest, and the action warns when a live digest differs from the recorded one.
