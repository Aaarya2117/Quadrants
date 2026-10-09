# Model Swap: Product Requirements Document

**Version:** 0.2 (replaces the evaluation-only v0.1)
**Track:** PS 04, Best Open-Source AI Project
**Status:** Draft

## 1. Overview

Model Swap is an open-source tool that replaces an open-weight model in a running application with a new one, safely and reversibly. It compares the model currently in production with a candidate on the team's own workload, presents the swap options with their trade-offs, and applies the chosen option by changing the model reference in the codebase. Every swap is verified, recorded, and can be rolled back in one command.

## 2. Problem

- Teams switch models to cut cost, improve privacy, or adopt a better release. The switch is usually a manual edit scattered across the codebase: hardcoded model names, prompts tuned for the old model, and output parsers that assume its format.
- Nobody has a clear picture of how the candidate differs from the current model on the team's real requests before the change goes out.
- When a swap goes wrong, rollback means finding and reverting several edits under pressure.

## 3. Goals

| ID | Goal | Measure |
|----|------|---------|
| G1 | Compare the production model and a candidate on the team's own workload | Side-by-side comparison report produced by one command |
| G2 | Present the swap options with trade-offs | Options list with risk level, expected metric changes, and required steps |
| G3 | Apply a swap by changing the model reference in the codebase | Swap changes only the model reference; the diff is shown and reviewable |
| G4 | Keep the application running through the swap | Application code outside the reference is untouched; the new model passes the contract checks before it is applied |
| G5 | Make every swap reversible | `rollback` restores the previous reference in one command, verified by test |
| G6 | Record every decision | Each swap writes an audit record with the comparison, the chosen option, and the approver |

## 4. Non-Goals (Hackathon Scope)

- Hosting, serving, or load-balancing models
- Downloading or training model weights as part of the swap (the tool points at weights, it does not manage their storage)
- Automatic traffic splitting in production (canary is a stretch goal)
- Rewriting prompts automatically for the new model
- Judging open-ended creative quality

## 5. Target Users

- **Application engineers** who want to change models without hunting for hardcoded references.
- **Platform and ML engineers** who run open-weight models on their own infrastructure and need controlled upgrades.
- **Teams with compliance or cost constraints** who must justify and record each model change.

## 6. Core Concepts

### 6.1 Model reference
Application code requests a role, such as `invoice_extractor`, not a model name. A registry file (`models.yaml`) maps each role to a model reference. The reference includes the backend, model tag, quantization, and content digest. Changing a role's model means changing one entry in that file.

### 6.2 Two swap paths
- **Registry path (preferred):** application code already uses roles. The swap updates `models.yaml` only.
- **Source path (legacy code):** code has hardcoded model strings. The tool finds them, shows the proposed edits as a diff, and applies them after approval. The tool does not rewrite code it cannot locate with certainty.

### 6.3 Contract checks
A candidate can only take a role if it passes the role's contract: the tool schemas, output format, and required fields that the calling code depends on. The contract is written once per role in `contracts/<role>.yaml`.

### 6.4 Swap options
After comparison, the tool presents options such as:
1. **Full swap:** switch the role to the candidate immediately.
2. **Keep current:** no change; report why the candidate was not chosen.
3. **Quantized variant:** use a smaller quantization of the same candidate, if it still passes.
4. **Staged swap (stretch):** route a fraction of requests to the candidate first.

Each option lists the expected changes in quality, cost, and latency, the risk level, and the steps the tool will perform.

## 7. Functional Requirements

### FR-1 Roles and registry
- FR-1.1 `models.yaml` maps role names to model references with fields: `backend`, `model`, `quantization`, `digest`, `params` (for example, temperature and max tokens).
- FR-1.2 Application code loads a model through `modelswap.get("role")`, which reads the registry at startup and returns a client.
- FR-1.3 A role's current reference and its previous reference are both stored, so rollback has a target.

### FR-2 Comparison
- FR-2.1 `modelswap compare --role invoice_extractor --candidate <ref>` replays the role's recorded requests through the current and candidate references.
- FR-2.2 Requests come from a suite file that is either hand-written or imported from logs with PII sanitization.
- FR-2.3 Comparison checks the contract first and then measures accuracy, format validity, cost per 1,000 requests, and p95 latency.
- FR-2.4 Results are paired per request, and differences are reported with confidence intervals.

### FR-3 Options
- FR-3.1 `modelswap options` lists the swap options for the comparison, each with risk level, expected changes, and steps.
- FR-3.2 A candidate that fails the contract is listed only as "keep current," with the failing contract checks.
- FR-3.3 Risk level is set by policy (`policy.yaml`), for example: high when the contract passes but accuracy drops with significance; medium when cost or latency worsens beyond budget; low when all metrics are within tolerance.

### FR-4 Apply
- FR-4.1 `modelswap apply --option <id>` shows the diff of changes, asks for confirmation (or `--yes` in CI), and applies it.
- FR-4.2 Registry path: updates the role's entry in `models.yaml` and moves the old entry to `previous`.
- FR-4.3 Source path: applies the edits found by the source scanner, only where the match is unambiguous.
- FR-4.4 After applying, the tool runs a smoke test on the role's contract cases against the new reference, and rolls back automatically if the smoke test fails.
- FR-4.5 Every apply writes an audit record to `swaps/<timestamp>.json`: the comparison summary, the option chosen, the diff, the approver, and the smoke-test result.

### FR-5 Rollback
- FR-5.1 `modelswap rollback --role <role>` restores the `previous` reference.
- FR-5.2 Rollback writes its own audit record and runs the same smoke test.
- FR-5.3 Rollback works on the registry path and the source path.

### FR-6 Source scanner (legacy path)
- FR-6.1 Scans the repository for model string literals and known client calls (for example, `model="..."` and `ModelClient(...)`).
- FR-6.2 Reports each match with file, line, and confidence. Only high-confidence matches are eligible for automatic edits.
- FR-6.3 Never edits files outside the configured paths.

### FR-7 Reports and CI
- FR-7.1 Markdown comparison and options report for pull requests.
- FR-7.2 Static HTML report with side-by-side metrics and the options table.
- FR-7.3 GitHub Action: on a pull request that changes `models.yaml` or a candidate file, runs comparison and posts the options as a comment. Applying a swap is a manual step, a workflow dispatch with the chosen option.

### FR-8 Safety
- FR-8.1 Secrets come from environment variables only and never appear in the registry, audit records, or reports.
- FR-8.2 A swap cannot be applied while a comparison is incomplete or the contract check failed.
- FR-8.3 Every apply and rollback is blocked unless the working tree is clean or the change is the only diff.

## 8. Non-Functional Requirements

- **Reversibility:** every swap can be undone in one command, tested in CI.
- **Transparency:** every change is a visible diff, and nothing is applied silently.
- **Portability:** Python 3.11+, Linux, macOS, and Windows via WSL.
- **Performance:** comparing 40 cases on one local model completes within 10 minutes on a consumer GPU.
- **Licensing:** Apache-2.0.

## 9. Success Metrics (Hackathon)

- A full cycle runs on the demo project: compare, options, apply, smoke test pass, rollback, audit records.
- A deliberately broken candidate is rejected at the contract stage and never applied.
- The source path applies a swap to a demo project with three hardcoded references and leaves all other lines unchanged.
- The rollback restores the original files byte-for-byte.

## 10. Milestones (48 Hours)

| Window | Milestone |
|--------|-----------|
| Hours 0–6 | Repo, `models.yaml` schema, `modelswap.get()`, Ollama backend |
| Hours 6–14 | Contract checker, comparison engine with paired metrics |
| Hours 14–22 | Options and policy risk levels, apply on registry path, smoke test |
| Hours 22–30 | Rollback, audit records, source scanner (high-confidence matches only) |
| Hours 30–38 | Source path apply, HTML and Markdown reports, GitHub Action comment |
| Hours 38–44 | Demo project with three roles, Gemma 4 multimodal role, seeded bad candidate |
| Hours 44–48 | Polish, rehearsal, pitch |

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Source scanner misses or misreads a reference | Only high-confidence edits are automatic; low-confidence matches are listed for manual change |
| Passing contract checks does not guarantee quality | Comparison metrics are shown before apply; the options show risk, not a guarantee |
| Prompts tuned for the old model degrade the new one | Contract and comparison cover this; prompt rewriting is a non-goal |
| Model weights are large and not stored in the repo | Registry stores a digest and pull reference; the tool verifies the digest before applying |
| Time pressure | Cut order: source path first, then staged swap; registry path, contract check, and rollback are protected |

## 12. Future Work

- Staged and canary swaps with automatic promotion and rollback on metric regression
- Weight pulling and digest verification integrated with model registries
- Automatic prompt adaptation proposals for the candidate, reviewed before apply
- Integration with serving platforms to reload models without restarting the application
