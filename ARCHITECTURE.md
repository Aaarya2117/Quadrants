# Model Swap: Architecture

## 1. Design Principles

1. **Application code depends on roles, not models.** The model reference lives in one registry file, so a swap is a config change, not a code change.
2. **Nothing changes silently.** Every swap is a visible diff, approved by a person, recorded in an audit file, and reversible.
3. **Contracts before comparisons.** A candidate that breaks what the calling code depends on is rejected before any metric is compared.
4. **Deterministic decisions.** Contract checks, statistics, and risk rules are code. Models produce outputs under test; they do not decide the swap.
5. **Legacy code is supported with care.** Hardcoded references are migrated only where the match is certain.

## 2. System Context

```
 +-------------------+      +--------------------------------------+
 | Engineer / CI     | ---> | modelswap CLI / GitHub Action       |
 +-------------------+      +--------------------------------------+
                               |            |             |
              +----------------+            |             +--------------+
              v                             v                            v
 +----------------------+    +-----------------------+    +-----------------------+
 | Repository           |    | Model endpoints       |    | Audit and run store   |
 | models.yaml          |    | Ollama, OpenAI-compat |    | runs/, swaps/         |
 | contracts/, src/     |    +-----------------------+    +-----------------------+
 +----------------------+
              ^
              | application calls get("role")
 +----------------------+
 | Application          |
 +----------------------+
```

## 3. Component Overview

```
modelswap/
├── cli.py                  # Typer commands: compare, options, apply, rollback, scan
├── runtime/
│   ├── registry.py         # Loads models.yaml; get(role) returns a client
│   ├── client.py           # Role client: chat(messages, tools) with role params
│   └── backends/
│       ├── base.py         # Backend protocol
│       ├── ollama.py
│       └── openai_compat.py
├── contracts/
│   ├── spec.py             # Contract model (schema, tool names, required fields)
│   └── check.py            # Validates a candidate's outputs against the contract
├── suite/
│   ├── schema.py           # Recorded request format
│   ├── importer.py         # JSONL import with sanitization
│   └── redact.py           # PII scanning and masking
├── compare/
│   ├── runner.py           # Async replay, caching, concurrency limits
│   ├── graders.py          # Accuracy and format grading
│   ├── stats.py            # Paired differences, McNemar, bootstrap intervals
│   └── metrics.py          # Cost, p50/p95 latency
├── options/
│   ├── builder.py          # Builds swap options from the comparison
│   └── policy.py           # Risk levels from policy.yaml
├── swap/
│   ├── planner.py          # Produces the change set (registry or source edits)
│   ├── apply.py            # Applies the change set, runs smoke test, auto-rollback
│   ├── rollback.py         # Restores previous references
│   └── audit.py            # Writes swaps/<timestamp>.json
├── source/
│   ├── scanner.py          # Finds model references in code, with confidence
│   └── editor.py           # Applies high-confidence edits only
├── report/
│   ├── markdown.py
│   └── html.py
└── ci/action/              # Composite GitHub Action wrapper
```

## 4. Repository Layout (User Project)

```
my-app/
├── models.yaml             # Role to model reference registry
├── contracts/
│   └── invoice_extractor.yaml
├── suites/
│   └── invoices.yaml       # Recorded requests with expected results
├── policy.yaml             # Risk rules for options
├── src/                    # Application code using get("role")
├── runs/                   # Comparison outputs (generated)
└── swaps/                  # Audit records (generated, committed)
```

## 5. Core Data Model

```
Role ──< ModelRef (current, previous)
Role ──1 Contract
Suite ──< Case
Comparison = (Role, Candidate, Suite) ──< PairedResult (case, current_outcome, candidate_outcome)
Comparison ──< Option (id, action, risk, expected_changes, steps)
Option ──1 ChangeSet (registry edits, source edits, diffs)
Swap ──1 AuditRecord (comparison hash, option, diffs, approver, smoke_test_result)
```

Key identifiers:
- `ref_id`: `backend:model@digest` for a model reference
- `comparison_id`: hash of role, candidate ref, suite hash, contract hash, and policy hash
- `change_set_hash`: hash of the proposed edits, checked again at apply time so the applied change equals the reviewed change

## 6. Key Interfaces

### Role client

```python
def get(role: str, registry_path: str = "models.yaml") -> RoleClient: ...

class RoleClient:
    role: str
    ref: ModelRef

    def chat(self, messages: list[Message], tools: list[Tool] | None = None) -> Response: ...
```

### Backend

```python
class Backend(Protocol):
    name: str

    def generate(self, ref: ModelRef, req: Request) -> Response:
        """Returns text, tool_calls, usage, latency_ms, cost estimate."""
```

### Contract check

```python
def check(contract: Contract, response: Response) -> ContractResult:
    """Returns passed, list of violations (field, rule, actual, expected)."""
```

### Change set

```python
class ChangeSet(BaseModel):
    registry_edits: list[RegistryEdit]   # role, from_ref, to_ref
    source_edits: list[SourceEdit]       # file, line, old, new, confidence
    hash: str                            # checked at apply time
```

## 7. Comparison Flow Internals

- The runner replays each suite case against the current and candidate references with the role's params.
- Responses are cached by (ref_id, prompt hash, params, case id), so re-running an option review is cheap.
- Contract checks run first. A failed contract stops that candidate and marks it as "keep current."
- Accuracy and format results are paired per case. McNemar's test covers binary outcomes; a paired bootstrap gives intervals on accuracy, cost, and latency differences.

## 8. Swap Mechanics

### Registry path
1. Planner reads `models.yaml` and produces a ChangeSet: `current` becomes the candidate, and the old `current` becomes `previous`.
2. Apply writes the file atomically (temporary file, then rename).
3. Smoke test runs the contract cases against the new `current`.
4. If the smoke test fails, apply restores the file from the saved copy and records the failure.

### Source path
1. Scanner finds model references and assigns a confidence score. Only matches above the threshold are eligible.
2. Planner builds source edits that change the reference to `get("role")` or to the new model string, depending on the mode chosen.
3. Apply checks that each target line still matches the expected old text (guards against drift since the scan).
4. Apply runs in a clean git state, or in a branch the tool creates, so the diff is reviewable and revertible.
5. Smoke test and automatic rollback follow the registry path.

### Rollback
- Reads the previous reference from `models.yaml`, or the reverse of the last audit record for source edits.
- Writes its own audit record and runs the smoke test.

## 9. Policy and Risk

`policy.yaml` defines how options are rated:

```yaml
risk:
  high:
    when: accuracy_drop_significant
  medium:
    when: cost_or_latency_over_budget
  low:
    when: all_within_tolerance
budgets:
  p95_latency_increase_pct: 30
  cost_increase_pct: 0
alpha: 0.05
```

Risk is computed from the comparison results and the policy, not from model output.

## 10. Safety

- Secrets come from environment variables. The config loader rejects inline keys.
- Audit records and reports contain no secrets and only masked recorded inputs.
- Apply refuses to run when the comparison is incomplete, the contract failed, or the change set hash no longer matches the reviewed one.
- The source editor touches only the files listed in the scan configuration.

## 11. Technology Choices

| Concern | Choice | Reason |
|---------|--------|--------|
| Language | Python 3.11+ | Common in ML tooling; contributors can start quickly |
| CLI | Typer | Typed commands with minimal boilerplate |
| Config | Pydantic v2, YAML | Validation with clear errors |
| HTTP | httpx (async) | Connection pooling and async support |
| Source edits | libcst or tree-sitter | Edits code structure rather than raw text, reducing mistakes |
| Storage | JSON files in the repository | Audit trail that travels with the code |
| Statistics | NumPy, SciPy | Standard paired tests and bootstrap |
| CI | GitHub composite Action | Works without a hosted service |

## 12. Known Risks

- **Scanner false positives or negatives.** Mitigated by confidence thresholds and the drift guard before apply.
- **Quality differences not covered by the suite.** Stated in the README; the suite is the team's responsibility.
- **Weights not present locally.** The digest check fails the swap rather than applying an unverified reference.
- **Prompt drift.** Not handled automatically; surfaced by the comparison, fixed by a person.
