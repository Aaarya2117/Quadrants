# Model Swap: Round 1 Idea Presentation

**Track:** Best Open-Source AI Project (PS 04: Build the AI System Behind the AI)
**Team:** Arjun, Bhagat, Achyut, Anirudh
**Status at Round 1:** Design complete. Implementation starts at the Hack Day. Anything not yet built is marked *Planned*.

---

## 1. Problem Statement & Proposed Solution

**Problem.** Teams that run open-weight models in production switch models for cost, privacy, or quality reasons. Each switch is usually a manual edit across the codebase: hardcoded model names, prompts tuned for the old model, and parsers that assume its output format. Nobody knows how the new model differs on the team's real requests until users report problems, and rolling back means reverting several edits under pressure.

**Who experiences it.** Application engineers who change models, platform and ML engineers who run open-weight models on their own infrastructure, and teams with cost or compliance rules that require each model change to be justified and recorded.

**Why it matters.** Open-weight models are cheaper and more private than hosted APIs, so teams switch more often. Without a safe process, each switch is a risk.

**Proposed solution.** Model Swap is an open-source tool that replaces a model in a running application safely and reversibly. Application code asks for a role (for example, `invoice_extractor`) instead of a model name, and a `models.yaml` registry maps that role to a model. The tool:

1. Compares the current model with a candidate on the team's recorded requests.
2. Checks each candidate against the output contract the code depends on.
3. Presents swap options with risk levels and expected changes.
4. Applies the chosen option by changing the model reference, after showing the diff.
5. Runs a smoke test, rolls back automatically on failure, and writes an audit record.

**How it differs from existing solutions.** Evaluation frameworks report scores. Model Swap goes from the comparison to a controlled change in the codebase, with contract checks, a diff for approval, automatic rollback, and an audit trail. It also works on legacy code, where it finds hardcoded model references and migrates only the high-confidence matches.

---

## 2. Dataset & AI Model

**Models.**
- Candidate and current models: open-weight models served through Ollama, including Gemma 4 for the multimodal role.
- No LLM judge. Contract checks and grading are deterministic, so the swap decision does not depend on a second model's opinion.

**Why these models.** Gemma 4's text and vision capabilities fit the Best Use of Gemma 4 requirement, and Ollama lets the tool run every model locally with open weights.

*Status:* The exact Gemma 4 tag and its vision support through Ollama must be verified before the event. Fallback: a Gemma 3 vision model.

**Datasets.**
- A demo project with three roles: an invoice extractor (Gemma 4, multimodal), a support tool-call router, and a ticket-summary role.
- Synthetic data only, about 15 to 40 recorded requests per role, with expected results.
- Optional import of the team's own logs in JSONL, with PII masking before storage.

**How AI contributes.** The models being compared are the subject of the tool. The swap logic, contract checks, and statistics are deterministic code. The models produce outputs; the tool decides whether and how to change the reference.

**Alignment with the track.** The tool is built for open-weight models and runs them locally. It is open-source under Apache-2.0.

---

## 3. Technology Stack & Architecture

**Stack.**
- Python 3.11+, Typer (CLI), Pydantic v2 with YAML, httpx (async), NumPy and SciPy (statistics)
- Ollama for local models; any OpenAI-compatible endpoint as a second backend
- libcst or tree-sitter for safe source edits
- JSON files for audit records, stored in the repository
- GitHub composite Action for pull-request comments

**Architecture.** The application calls `get(role)`. The registry resolves the role to a model reference. A separate swap engine works on the registry and, when needed, on source files. The main components:

1. **Runtime:** registry loader, role client, and backends.
2. **Comparison:** recorded-suite replay with caching, contract checks, graders, and paired statistics.
3. **Options and policy:** builds swap options and assigns risk levels from `policy.yaml`.
4. **Swap engine:** planner, atomic apply, smoke test, automatic rollback, and audit records.
5. **Source scanner:** finds hardcoded references with confidence scores and applies only high-confidence edits on a branch.
6. **Reports and CI:** Markdown and HTML reports, and a GitHub Action that comments on pull requests.

**Key design choices.**
- *Roles, not model names, in application code:* a swap changes one file, not the code.
- *Contract before comparison:* a candidate that breaks the output format is rejected before any metric is compared.
- *Model reference, not model weights, in the repository:* weights are large binaries. The registry stores the model and its digest, and the tool verifies the digest before applying.
- *Human approval and automatic rollback:* no swap is applied silently, and a failed smoke test restores the previous state.

**Status.** *Planned:* all components. The design is documented in `ARCHITECTURE.md` and `FLOW.md`.

---

## 4. Novelty & Innovation

**Key innovation: the path from comparison to a safe change.** Most model comparison tools stop at a report. Model Swap uses the comparison to choose a swap option, applies that option to the codebase as a reviewable diff, verifies it with a smoke test, and records it. Rollback is one command.

**Second innovation: contract-gated swaps.** Each role has a contract that describes the tool names, required fields, and output schema the calling code depends on. A candidate has to pass it before it can be considered.

**Third innovation: migration of legacy code.** The source scanner finds hardcoded model references and rates each match by confidence. Only high-confidence matches are edited automatically, which lets teams adopt the tool without rewriting their code first.

**Type of contribution.** A new capability that combines model comparison, controlled change, and rollback. It improves an existing manual process rather than replacing an evaluation tool.

---

## 5. Implementation Plan & Demonstration

**Core features (MVP, in priority order).**
1. Model registry and `get(role)` with the Ollama backend
2. Recorded suite format and comparison (contract checks, paired metrics)
3. Options with risk levels from `policy.yaml`
4. Registry apply, smoke test, and automatic rollback
5. Rollback command and audit records
6. Markdown and HTML comparison report
7. GitHub Action that posts the options as a pull-request comment
8. Gemma 4 multimodal role in the demo project
9. Source scanner and source-path apply on a git branch (high-confidence matches only)
10. OpenAI-compatible backend

**How we will demonstrate it.**
- Run a comparison on the demo project: the candidate fails the contract on one role and passes on another.
- Show the options table: "keep current" for the failing candidate, a full swap with low risk for the passing one.
- Apply the swap, show the diff, and show the smoke test passing.
- Roll back and confirm the original file is restored byte for byte.
- Show a legacy file with hardcoded references, scanned and migrated on a branch.
- Keep a pre-recorded backup video in case the live demo fails.

**What is realistic in 48 hours.** Items 1 to 5 and 6 in basic form are the core. Items 7 and 8 are targets. Items 9 and 10 are stretch goals, cut first if the schedule slips.

**Division of work.**
- Arjun: runtime, swap engine, source scanner, CLI
- Bhagat: recorded suites, contracts, comparison runner, demo project
- Achyut: statistics, options, risk policy, digest verification
- Anirudh: reports, GitHub Action, README, pitch

---

## 6. Feasibility & Real-World Impact

**Can it be built in the time?** The core scope, meaning registry, comparison, options, apply, and rollback, is achievable in 48 hours if the team holds the checkpoints in `WORK_DIVISION.md`. The swap engine is the riskiest part because it writes files, so it is scheduled early and tested with a byte-for-byte rollback check.

**Who benefits.** Small product teams and internal platform teams that run open-weight models and change them regularly, and organisations that must record why each model change was made.

**Real-world impact.** Fewer model-change incidents, faster and safer rollbacks, and an audit trail for every change.

**Limitations and risks.**
- Passing the contract and comparison lowers risk; it does not guarantee quality on inputs the suite does not cover.
- Prompts tuned for the old model may need manual adjustment. The tool reports this but does not rewrite prompts.
- The source scanner may miss references or misclassify them. It only edits high-confidence matches and drift-checks each line before editing.
- The demo uses synthetic data, so it shows the method, not production accuracy.
- Gemma 4 availability and vision support through local tooling must be verified before the event.
- Staged or canary swaps are planned for later, not built during the hackathon.

---

## 7. Challenge Requirements

**Best Use of Gemma 4.** Gemma 4 runs the multimodal invoice-extraction role in the demo. The demo can swap between Gemma 4 and another open-weight candidate, and show the contract and comparison results for each.

**Best Open-Source AI Project.**
- The tool is open-source under Apache-2.0.
- Every model in the demo is open-weight and runs locally through Ollama or a compatible server.
- The final project will be in a public GitHub repository with a README covering what it does, how to run it, and the models and dependencies used.

---

## 8. Clarity & Team Collaboration

**One-sentence pitch.** Model Swap lets you change the open-weight model behind your application safely: compare it on your own requests, apply the swap as a reviewable change, and roll back in one command.

**How the team will explain it.**
- Start with the problem: a model change that is a one-line idea on paper turns into a risky multi-file edit.
- Show the flow: compare, options, apply, smoke test, rollback.
- Close with the demo: a candidate rejected by the contract, and a candidate applied and rolled back.

**Team roles.** Each member can explain the full flow. Each person presents their own area and can answer questions on the others'.

**Honesty about status.** Everything in this document is design. Nothing is built yet. Demo results will come from runs on the day, and limits will be stated as they are.

---

## Action Checklist

### Before Round 1 (Mentor Discussion)

- [ ] Assign a presenter for each section: problem, models, architecture, novelty, plan, feasibility, challenge fit, Q&A
- [ ] Rehearse a 5-minute pitch and a 5-minute question round
- [ ] Prepare a one-page architecture diagram from `ARCHITECTURE.md`
- [ ] Prepare a one-page swap flow diagram from `FLOW.md` (compare, options, apply, smoke test, rollback)
- [ ] Verify Gemma 4 can be pulled and run through Ollama on at least one laptop, and record the exact tag
- [ ] Confirm MLH portal registration is complete for all four members
- [ ] Prepare answers for: what is built, what is planned, what is uncertain
- [ ] Prepare answers for: "why not change the weights in the repo?" (size, sharing, digest verification)
- [ ] Prepare answers for: "how is this different from evaluation frameworks?" (it applies and rolls back the change)
- [ ] Bring a shared copy of `WORK_DIVISION.md` and `TEAM_ASSIGNMENTS.md`

### Setup on Day 1 (Hours 0–2)

- [ ] Create the public GitHub repository with the Apache-2.0 license
- [ ] Add the folder layout from `ARCHITECTURE.md`
- [ ] Agree and commit the shared types in `contracts.py`
- [ ] Set up linting and tests in CI
- [ ] Install Ollama and pull the candidate and current models on every laptop
- [ ] Record the model tags and digests for the demo

### Build Milestones

- [ ] CP1 (Hour 6): `get("role")` calls the model; `models.yaml` loads
- [ ] CP2 (Hour 14): comparison on 15 cases produces paired results and a contract verdict
- [ ] CP3 (Hour 22): options list shows risk; bad candidate is "keep current"
- [ ] CP4 (Hour 30): registry apply passes smoke test; rollback restores the file byte for byte
- [ ] CP5 (Hour 38): source path applies on the demo branch; PR comment posts from the action
- [ ] CP6 (Hour 44): full cycle passes twice; feature freeze

### Repository and Documentation

- [ ] README matches implemented behaviour; remove any feature that does not work
- [ ] Create the linked files or remove the links: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`
- [ ] Verify every Ollama model tag in the quick start
- [ ] Replace any illustrative numbers with results from real runs
- [ ] Commit the demo project, its recorded suites, and a sample audit record
- [ ] Add a privacy note confirming the demo data is synthetic

### Demo Preparation

- [ ] Demo project set up with the GitHub Action enabled
- [ ] Full cycle rehearsed twice: compare, options, apply, smoke test, rollback
- [ ] Backup screen recording made
- [ ] Known failure modes listed, with what the team will say if they happen

### Final Hours (44–48)

- [ ] Bug fixes only; no new features after Hour 44
- [ ] Install test on a clean machine
- [ ] One-page explanation of options and risk rules for judges
- [ ] Each member explains the flow aloud once
- [ ] Release tagged; README checked against the tag

### Judging Day

- [ ] Open with the problem and the one-sentence pitch
- [ ] Walk through the options table and the rollback
- [ ] State what is built, what was cut, and what is planned
- [ ] Point out the Gemma 4 role and the open-weight models in the demo
- [ ] Record mentor feedback for follow-up after the event
