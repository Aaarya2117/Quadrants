# Model Gate: Round 1 Idea Presentation

**Track:** Best Open-Source AI Project (PS 04: Build the AI System Behind the AI)
**Team:** Arjun, Bhagat, Achyut, Anirudh
**Status at Round 1:** Design complete; implementation starts at the Hack Day. Items are marked *Planned* unless stated otherwise.

---

## 1. Problem Statement & Proposed Solution

**Problem.** Teams choose open-weight models from leaderboard scores. Those scores do not reflect the team's own prompts, tool schemas, JSON contracts, latency targets, or cost limits. A model swap that looks like a saving can quietly break production behavior: malformed tool calls, invalid JSON, or lower accuracy on one workflow. Teams usually find out after users complain.

**Who experiences it.** Platform and ML engineers who maintain model configs, application engineers who change prompts or tool schemas, and teams in cost-sensitive or regulated settings who must justify model changes.

**Why it matters.** Open-weight models are cheaper and more private than hosted APIs, so teams are switching more often, and each switch carries regression risk. Without a check, the switch is a guess.

**Proposed solution.** Model Gate is an open-source CLI and CI gate. It replays a team's own recorded requests through a baseline model and a candidate model, grades the outputs deterministically, applies statistical tests to decide whether a difference is real, and returns a merge recommendation (pass, block, or inconclusive) with a saved evidence bundle.

**How it differs from existing solutions.** Generic eval frameworks run a fixed benchmark or a single pass/fail count. Model Gate focuses on migration decisions: it compares two models on the same cases, separates format failures from reasoning failures, uses sequential statistics to stop early, and treats cost and latency budgets as part of the gate.

---

## 2. Dataset & AI Model

**Models.**
- Primary: Gemma 4 for the multimodal task (image plus text extraction) and as one of the candidate models in the demo.
- Baseline for comparison: a Gemma-family model served through Ollama.
- Judge: no LLM judge in the MVP. All grading is deterministic, so results do not depend on a second model's opinion.

**Why these models.** Gemma 4's text and vision capabilities fit the Best Use of Gemma 4 requirement directly. Running through Ollama or an OpenAI-compatible endpoint keeps the tool model-agnostic and lets teams use open weights on their own hardware.

*Status:* Gemma 4 model tag and vision support through Ollama need to be verified on the day. Fallback is a Gemma 3 vision model if Gemma 4 is not available locally.

**Datasets and data sources.**
- A demo task suite of about 40 hand-written cases: 20 tool-call cases, 15 JSON-extraction cases, and 5 multimodal cases (invoices, forms, and notices).
- Synthetic data only. No customer or personal data is used.
- Optional import path for teams' own traces in JSONL, with a PII scan before anything is stored.

**How AI contributes to core functionality.** The model being evaluated is the subject of the tool, not a helper inside it. The tool's logic (grading, statistics, policy) is deterministic code. AI produces the outputs under test; the harness decides whether those outputs are good enough.

**Alignment with the track.** Open-weight models are the centre of the workflow: the tool exists to make open-model migration safe, and it runs entirely with open-weight models locally.

---

## 3. Technology Stack & Architecture

**Stack.**
- Language: Python 3.11+
- CLI: Typer
- Config and validation: Pydantic v2, YAML
- Async HTTP: httpx
- Cache and storage: SQLite, JSON bundles
- Statistics: NumPy, SciPy
- Reports: Markdown templates, a static HTML page with a Pareto chart
- CI: GitHub composite Action
- Model serving: Ollama for local runs; any OpenAI-compatible endpoint as the second backend

**Architecture.** Six layers, each with a single job:

1. **Config and suite loader** validates YAML task suites and the gate policy.
2. **Planner and runner** expands each case across models, seeds (K=3), and decoding modes, then executes calls asynchronously with per-provider limits and caching.
3. **Adapters** wrap each backend behind one interface: `generate()` and `capabilities()`.
4. **Graders** check outputs against expected results and assign a failure class.
5. **Stats and policy** run paired tests and sequential monitoring, then evaluate rules in `gate.yaml`.
6. **Outputs** write the bundle, Markdown and HTML reports, and the exit code. The GitHub Action posts the PR comment and commit status.

**Key design choices and reasons.**
- *Deterministic graders, no LLM judge in the MVP:* avoids a second model's bias and keeps results reproducible.
- *Trace replay with stored responses:* makes CI cheap, and a decision can be replayed later with zero model calls.
- *Free and constrained decoding compared side by side:* shows whether a failure is a format problem or a reasoning problem, which points to different fixes.
- *Sequential statistics:* stops early when the evidence is clear and returns "inconclusive" otherwise, so the gate does not guess on small samples.

**Status.** *Planned:* all components. Architecture and interfaces are documented in `ARCHITECTURE.md`.

---

## 4. Novelty & Innovation

**Key innovative approach.** Diagnosis that separates constraint failures from semantic failures. Each structured case is run under free decoding and under grammar-constrained decoding. The combined result sorts every failure into one of four cases: healthy, format-only, semantic, or constraint-conflict. This tells the team which fix to apply, rather than only reporting a lower accuracy number.

**Second innovation.** Anytime-valid sequential decisions for model migration. The gate uses early-stopping statistics with controlled false-block rates, so compute is spent only where the decision is unclear.

**Third innovation.** Decision on the Pareto frontier. The policy compares candidates on quality, cost, and latency together, so a cheaper model is not accepted just because it is cheaper.

**Type of contribution.** We are introducing a new capability (migration gating with diagnosis and sequential statistics), not only improving an existing eval process.

---

## 5. Implementation Plan & Demonstration

**Core features (MVP, in priority order).**
1. Ollama adapter and CLI `run` command
2. Task suite format with `tool_call` and `json_extract` cases
3. Deterministic graders and failure taxonomy
4. Paired statistics (McNemar and bootstrap) and the `gate.yaml` policy
5. Markdown report with the decision and failure classes
6. Reproducibility bundle and `replay` command
7. GitHub Action that posts a PR comment and commit status
8. Gemma 4 multimodal case
9. Constrained-decoding diagnosis (only if the backend supports it)
10. Fix-hint loop (template-based)
11. Pareto view and HTML report

**How we will demonstrate it.**
- Live run on a demo repository: change the candidate model in a pull request.
- The gate blocks the change because tool-call accuracy drops with statistical significance, and the report names the failing case clusters.
- Replay the saved bundle with no model calls and show the same decision.
- Show the Gemma 4 image case passing or failing with clear reasons.
- A pre-recorded backup video in case the live demo fails.

**What is realistic in 48 hours.** Items 1 to 7 and item 8 in a basic form. Items 9 to 11 are stretch goals and will be cut in the order listed in `WORK_DIVISION.md` if the schedule slips.

**Division of work.**
- Arjun: CLI, adapters, runner, cache, bundle and replay
- Bhagat: task suite, graders, taxonomy, multimodal case, seeded regression
- Achyut: statistics, sequential gate, policy engine, Pareto view, fix hints
- Anirudh: reports, GitHub Action, demo repository, README and pitch

---

## 6. Feasibility & Real-World Impact

**Can it be built in the time?** Yes for the MVP scope, provided the team freezes features at Hour 44 and follows the checkpoints in `WORK_DIVISION.md`. The largest risks are the statistics layer and the GitHub Action integration, so both are scheduled early with tests.

**Who benefits.** Any team that runs open-weight models in an application and changes models or prompts regularly: small product teams, internal platform teams, and organisations with compliance needs for model changes.

**Practical impact.** Fewer silent regressions after model changes, a measured cost-quality trade-off in place of guesses, and an audit trail of why a model was approved.

**Limitations and risks.**
- The tool is only as good as the test suite. A suite that does not match real traffic gives misleading results.
- Constrained decoding depends on the backend. Some servers do not support it, so the diagnosis is partial for those.
- Sequential statistics need enough cases to reach a decision; small suites will often return "inconclusive."
- The demo uses synthetic data, so the results show the method, not real-world accuracy.
- Gemma 4 availability and vision support through local tooling must be verified before the event.

---

## 7. Challenge Requirements

**Best Use of Gemma 4.** Gemma 4 is one of the models under test and is used directly for the multimodal extraction task. The demo compares Gemma 4 variants (for example, full precision against a quantized build) and shows where each one fails.

**Best Open-Source AI Project.**
- The core tool is open-source under Apache-2.0.
- Open-weight models are central: every model in the demo is open-weight and runs locally through Ollama or a compatible server.
- The final project will be published in a public GitHub repository with a README covering what the project does, how to run it, and the models and dependencies used.

---

## 8. Clarity & Team Collaboration

**One-sentence pitch.** Model Gate tells you whether a cheaper open-weight model is safe to ship for your own workload, before you merge the change.

**How the team will explain the idea.**
- Lead with the failure: a leaderboard-backed model swap cuts cost but drops tool-call accuracy on real requests.
- Show the four-part diagnosis (healthy, format-only, semantic, constraint-conflict) and what each one means for the fix.
- Close with the gate decision and the replay.

**Team roles and understanding.** Every member can explain the full flow, not only their own part. Each member has presented their folder to the others in the design review.

**Honesty about status.** Everything in this document is design. Nothing is implemented yet. Results shown during the demo will come from runs on the day, and any limits will be stated as they are.

---

## Action Checklist

### Before Round 1 (Mentor Discussion)

- [ ] Confirm the presenter for each section (problem, model, architecture, novelty, plan, feasibility, challenge fit, Q&A)
- [ ] Rehearse a 5-minute pitch and a 5-minute question round
- [ ] Prepare a one-page architecture diagram from `ARCHITECTURE.md`
- [ ] Prepare a one-page workflow diagram from `FLOW.md`
- [ ] Verify Gemma 4 can be pulled and run through Ollama on at least one team laptop, and note the exact model tag
- [ ] Confirm the MLH event portal registration is complete for all four members
- [ ] Prepare honest answers for: what is built, what is planned, what is uncertain
- [ ] Agree on the answer to "why not use an LLM judge?" (reproducibility and bias)
- [ ] Agree on the answer to "how is this different from existing eval frameworks?"
- [ ] Bring a printed or shared copy of the work division and checkpoints

### Setup on Day 1 (Hours 0–2)

- [ ] Create the public GitHub repository with the Apache-2.0 license
- [ ] Add the folder skeleton from `ARCHITECTURE.md`
- [ ] Agree and commit `contracts.py` (Request, Response, Case, Outcome, Decision, Rule)
- [ ] Set up linting and test runner in CI
- [ ] Install Ollama and pull the chosen baseline and candidate models on every laptop
- [ ] Choose the demo pair and record the exact model tags and quantization levels

### Build Milestones

- [ ] CP1 (Hour 6): adapter returns a response; suite loads; report renders from fake data
- [ ] CP2 (Hour 14): real 10-case run produces a graded report
- [ ] CP3 (Hour 22): policy blocks the seeded regression; Pareto view exists
- [ ] CP4 (Hour 30): Gemma 4 image case runs; early stop works
- [ ] CP5 (Hour 38): replay works with zero model calls; GitHub Action comments on the demo PR
- [ ] CP6 (Hour 44): full demo passes twice; feature freeze

### Repository and Documentation

- [ ] README matches the implemented behaviour (no features listed that do not work)
- [ ] Remove or create the linked files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `docs/ci-recorded-mode.md`, `examples/`
- [ ] Verify all Ollama model tags in the README quick start
- [ ] Replace illustrative numbers in the README and pitch with results from real runs
- [ ] Include the seeded regression suite and its saved bundle in the repository
- [ ] Add a privacy note stating that demo data is synthetic

### Demo Preparation

- [ ] Demo repository set up with the GitHub Action enabled
- [ ] Full demo rehearsed twice end to end
- [ ] Backup screen recording made in case of live failure
- [ ] Offline copy of the HTML report available
- [ ] Known failure modes listed, with what the team will say if they occur

### Final Hours (44–48)

- [ ] Bug fixes only; no new features after Hour 44
- [ ] Install test on a clean machine: `pip install` and the quick start both work
- [ ] One-page statistics explanation prepared for judges
- [ ] Team walkthrough: each member explains the flow aloud once
- [ ] Final commit tagged and README verified against the tagged version

### Judging Day

- [ ] Open with the failure example and the one-sentence pitch
- [ ] Be ready to show the diagnosis matrix and the Pareto view
- [ ] State clearly what is implemented, what was cut, and what is planned
- [ ] Note the Gemma 4 use and the open-weight models in the demo
- [ ] Capture mentor feedback and record follow-up items for after the event
