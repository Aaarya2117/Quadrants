# Model Gate: Product Requirements Document

**Version:** 0.1 (hackathon scope)
**Track:** PS 04, Best Open-Source AI Project
**Status:** Draft

## 1. Overview

Model Gate is an open-source CLI and CI gate that decides whether a candidate open-weight model can replace a baseline model for a specific team's workload. It replays recorded production-style requests through both models, separates format failures from reasoning failures, applies anytime-valid statistics so decisions can be made early and honestly, and outputs a Pareto-based merge recommendation with reproducible evidence.

## 2. Problem

- Teams pick models from leaderboards, which do not reflect their prompts, tool schemas, or output contracts.
- Migration regressions are often silent. A quantized or cheaper model can pass a casual test and still break tool calls in 3 percent of real requests.
- When regressions appear, teams cannot tell whether the model "cannot follow the format" or "reasons incorrectly." The fix is different in each case, so an aggregate score is not enough.
- Naive A/B comparisons either run too long or make false calls on small samples.

## 3. Goals

| ID | Goal | Measure |
|----|------|---------|
| G1 | Decide baseline vs candidate on the team's own workload | Gate decision produced for any two models with one command |
| G2 | Separate format failures from semantic failures | Every failure carries a class label (see section 6) |
| G3 | Make statistically honest decisions with less compute | Gate stops early when evidence is conclusive; false-block rate controlled at alpha |
| G4 | Treat quality, cost, and latency as trade-offs | Report shows a Pareto view; policy blocks on any violated budget |
| G5 | Be reproducible months later | Reproducibility bundle reproduces the same decision from stored artifacts |
| G6 | Integrate into existing pull-request workflow | GitHub Action posts a report and sets a required status |

## 4. Non-Goals (Hackathon Scope)

- Hosting or serving models
- A web dashboard beyond a static HTML report
- Training, fine-tuning, or quantizing models
- Replacing human evaluation for open-ended creative quality
- Real-time online traffic splitting

## 5. Target Users

- **Platform and ML engineers** who maintain model configs and need safe upgrades.
- **Application engineers** who change prompts or tool schemas and need regression checks.
- **Teams in regulated or cost-sensitive settings** who need an audit trail for model changes.

## 6. Core Differentiators

### 6.1 Trace replay, not synthetic-only tests
Teams export sanitized request records (messages, tool schemas, expected outputs or validators) from logs. Model Gate replays them offline against each model. No live traffic is touched.

### 6.2 Constraint-separated diagnosis
For structured outputs, each candidate is run twice:
1. **Free decoding:** the model's natural output, graded for correctness.
2. **Constrained decoding:** the same prompt with a grammar derived from the tool or JSON schema (GBNF in llama.cpp, guided decoding in vLLM, or Ollama's JSON schema format).

The combination yields four classes:

| Free decoding | Constrained decoding | Diagnosis |
|---------------|----------------------|-----------|
| Pass | Pass | Healthy |
| Fail, format | Pass | Format-only problem; fix with constraints or prompt |
| Fail | Fail | Semantic failure; the model reasons wrongly |
| Pass | Fail | Constraint conflicts with the model's preferred output; investigate schema |

This separates "the model is bad at following the format" from "the model is bad at the task," which drives different fixes.

### 6.3 Anytime-valid sequential decisions
Instead of a fixed sample size, the gate tests after each batch of paired cases using an always-valid method (e-values or a sequential paired test with alpha spending). It can:
- **Block early** when the regression is confirmed beyond the threshold.
- **Pass early** when non-inferiority is confirmed.
- **Continue** when the evidence is inconclusive, up to a maximum budget, then return "inconclusive," which the policy can treat as a block or a warning.

This keeps the false-block rate controlled and reduces compute on clear cases.

### 6.4 Pareto-based decision
The report plots quality against cost and against p95 latency. A candidate is recommended only if it is not dominated on the axes the policy cares about, or if it meets a stated quality floor within budget. This avoids the common mistake of optimizing one number.

## 7. Functional Requirements

### FR-1 Task suites
- FR-1.1 Tasks are defined in YAML with fields: `id`, `kind`, `input` (messages, images optional), `tools` or `schema` (optional), `grader`, `weight`.
- FR-1.2 Supported kinds at MVP: `tool_call`, `json_extract`, `multimodal_extract`.
- FR-1.3 Traces can be imported from JSONL with the same schema.

### FR-2 Adapters
- FR-2.1 Adapters implement `generate(request) -> Response` with usage, latency, and raw output.
- FR-2.2 MVP adapters: Ollama (local, including Gemma 4 vision where available) and one OpenAI-compatible hosted endpoint.
- FR-2.3 Constrained decoding is an optional adapter capability; adapters without it are reported as "free-decoding only."

### FR-3 Runner
- FR-3.1 Runs cases concurrently with per-provider concurrency limits.
- FR-3.2 Runs each case for K seeds (default 3).
- FR-3.3 Caches responses keyed by (model id, adapter version, prompt hash, seed, decoding mode).

### FR-4 Graders
- FR-4.1 `tool_call`: tool name in allowed set, required arguments present, argument types valid, no extra tools called.
- FR-4.2 `json_extract`: JSON parses, validates against schema, field-level exact or normalized match against expected values.
- FR-4.3 `multimodal_extract`: same as FR-4.2 with image input.
- FR-4.4 Every failure is assigned one taxonomy class (section 9).

### FR-5 Statistics
- FR-5.1 Paired per-case outcomes; McNemar test for binary outcomes; paired bootstrap for continuous metrics (latency, cost).
- FR-5.2 Sequential monitoring with alpha spending; decision states: `block`, `pass`, `continue`, `inconclusive`.
- FR-5.3 Report includes effect size and confidence interval, not only p-values.

### FR-6 Policy
- FR-6.1 `gate.yaml` defines rules, for example:
  ```yaml
  rules:
    - metric: tool_call_accuracy
      max_drop: 0.02
      confidence: 0.95
      action: block
    - metric: p95_latency_ms
      max_increase_pct: 30
      action: block
    - metric: cost_per_1k
      max_increase_pct: 0
      action: warn
  ```
- FR-6.2 Policy evaluation returns a machine-readable decision and a human-readable reason per rule.

### FR-7 Reporting
- FR-7.1 Markdown report for PR comments.
- FR-7.2 Static HTML report with Pareto chart, failure class breakdown, and example traces.
- FR-7.3 JSON output for other tools.

### FR-8 Fix hints
- FR-8.1 For each failure class with at least N occurrences, a template proposes a change (for example, add the tool schema to the system prompt, enable constrained decoding, reduce temperature).
- FR-8.2 The tool re-runs only affected cases with the proposed change applied and reports whether the failure class shrinks with significance.

### FR-9 Reproducibility
- FR-9.1 Each run writes a bundle: model identifiers and digests, quantization tag, adapter version, prompt hashes, seeds, grader versions, suite hash, policy hash, and raw responses.
- FR-9.2 `model-gate replay <bundle>` reproduces the decision from stored responses without calling models.

### FR-10 CI integration
- FR-10.1 GitHub composite Action with inputs: suite path, policy path, baseline and candidate configs.
- FR-10.2 Posts or updates a single PR comment.
- FR-10.3 Sets a commit status: success on `pass`, failure on `block`, neutral on `inconclusive` (configurable).

## 8. Non-Functional Requirements

- **Reproducibility:** identical inputs and stored responses produce identical decisions.
- **Cost control:** a `--max-spend` flag stops runs at a dollar limit.
- **Security:** API keys read from environment variables only; never written to bundles or logs; a redaction pass runs on traces before they are stored.
- **Privacy:** the README requires teams to sanitize traces; the tool warns on detected emails, phone numbers, and national ID patterns.
- **Portability:** Python 3.11+, runs on Linux, macOS, and Windows (WSL).
- **Licensing:** Apache-2.0.

## 9. Failure Taxonomy (MVP)

| Class | Definition |
|-------|-----------|
| `tool_name_wrong` | Called a tool not in the allowed set |
| `tool_args_missing` | Required argument absent |
| `tool_args_type` | Argument has the wrong type |
| `tool_extra_call` | Called a tool when none was needed |
| `json_invalid` | Output does not parse |
| `json_schema_violation` | Parses but fails the schema |
| `value_mismatch` | Valid structure, wrong extracted value |
| `truncated` | Output stopped at the token limit |
| `over_refusal` | Refused a benign case |
| `image_misread` | Multimodal value wrong where the image is clear (requires a human-labeled subset) |

## 10. Success Metrics (Hackathon)

- A full demo run on a 40-case suite completes in under 5 minutes on a single consumer GPU or a hosted endpoint.
- The gate catches a seeded regression (quantized candidate) and passes a known-good candidate.
- Decision is reproducible from the saved bundle with zero model calls.
- The PR comment and status appear in a live demo repository.

## 11. Milestones (48 Hours)

| Window | Milestone |
|--------|-----------|
| Hours 0–6 | Repo scaffold, adapter interface, Ollama adapter, task schema |
| Hours 6–14 | Runner, cache, tool_call and json_extract graders |
| Hours 14–22 | Stats module, sequential gate, policy evaluation |
| Hours 22–30 | Gemma 4 multimodal task, Pareto report, constrained-decoding probe |
| Hours 30–38 | GitHub Action, PR comment, bundle and replay |
| Hours 38–44 | Fix-hint step, demo suite, seeded regression |
| Hours 44–48 | README polish, demo rehearsal, pitch |

## 12. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Local models are slow in CI | Use cached recorded responses for PR runs; live runs on schedule |
| Constrained decoding unavailable on some backends | Mark as free-decoding only; report the gap |
| Judge bias if LLM-as-judge is used | Keep judge out of MVP; graders are deterministic |
| Small sample sizes | Sequential gate returns `inconclusive` rather than guessing |
| Scope creep | Freeze scope at the milestone table; anything else goes to "future" |

## 13. Future Work

- Shadow mode against live traffic with sampled logging
- Pluggable LLM-as-judge with calibration against human labels
- Pareto optimization across quantization levels automatically
- Provider marketplace of adapters
