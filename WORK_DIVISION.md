# Model Gate: Work Division

Team size: 4 (the plan also works for 2–3 members by merging roles, shown at the end).

## 1. Roles

| Role | Owner | Owns |
|------|-------|------|
| **A. Core and Runner** | Member 1 | CLI, config, adapters, executor, cache, bundle writer and replayer |
| **B. Tasks and Graders** | Member 2 | Suite schema, trace import and sanitizer, graders, taxonomy, Gemma 4 multimodal task, demo suite |
| **C. Stats and Policy** | Member 3 | Paired statistics, sequential gate, policy engine, Pareto computation, fix-hint loop |
| **D. Reporting, CI, and Demo** | Member 4 | Markdown and HTML reports, GitHub Action, demo repository, seeded regression, README, pitch |

Each member owns their folder under `model_gate/`. Cross-folder changes go through a short pull request reviewed by the owner of that folder.

## 2. Interface Contracts (Agree in Hour 0–1)

Before anyone writes logic, the team fixes these contracts in `model_gate/contracts.py`:

- `Request`, `Response` (text, tool_calls, usage, latency_ms, cost, raw)
- `Case`, `Suite`, `Expansion`
- `Outcome` (passed, failure_class, details, grader_version)
- `Decision` (pass | block | inconclusive, reasons list)
- `Rule` (from `gate.yaml`)

Owners:
- A writes `Response` and the adapter protocol
- B writes `Case`, `Outcome`, and the taxonomy enum
- C writes `Rule` and `Decision`
- D consumes all of them for reports

Freeze the contracts at Hour 1. Changes after that need all four members to agree.

## 3. Hour-by-Hour Plan (48 Hours)

### Day 1 (Hours 0–24)

**Hours 0–1: All members**
- Create the repo, license (Apache-2.0), and folder layout
- Agree on contracts (section 2)
- Set up CI linting and a test runner
- Choose the demo model pair (baseline: `gemma3:12b`; candidate: a q4 quantized variant of the same family, or a smaller model that is clearly weaker on tool calls)

**Hours 1–6**

| Member | Tasks | Done when |
|--------|-------|-----------|
| A | Typer CLI skeleton, config loader, Ollama adapter with `generate()` | `model-gate run --help` works; one Ollama call returns a `Response` |
| B | Suite YAML schema for `tool_call` and `json_extract`; 10 seed cases written by hand | Suite loads and validates; 10 cases reviewed |
| C | Implement McNemar and paired bootstrap with unit tests on synthetic data | Tests pass on known inputs |
| D | Report template (Markdown); HTML shell with placeholder chart; repo README skeleton | Sample report renders from fake data |

**Hours 6–14**

| Member | Tasks | Done when |
|--------|-------|-----------|
| A | Async executor with per-provider semaphore; SQLite cache keyed by `expansion_key` | Two identical runs: second run has zero model calls |
| B | `tool_call` and `json_extract` graders; taxonomy classes; trace importer with PII scan | Graders pass unit tests; importer flags emails and phone numbers |
| C | Sequential gate with alpha spending; decision states | Simulated stream of outcomes stops at the expected point |
| D | Markdown report wired to real `Outcome` objects; failure class table | Report from a real 10-case run looks correct |

**Hours 14–22**

| Member | Tasks | Done when |
|--------|-------|-----------|
| A | OpenAI-compatible adapter; constrained-decoding capability flags | Both adapters pass the same smoke test |
| B | Expand demo suite to 40 cases across tool calls and JSON; write `multimodal_extract` case format and image loader | 40 cases validated; one image case runs end to end |
| C | `gate.yaml` policy engine returning reasons; Pareto frontier computation | Policy blocks a seeded bad run; Pareto returns correct frontier on test points |
| D | HTML report with Pareto chart and diagnosis matrix | HTML opens offline and shows both charts |

**Hours 22–30 (Day 2 morning)**

| Member | Tasks | Done when |
|--------|-------|-----------|
| A | Constrained-decoding expansion in planner; diagnosis join logic | Diagnosis matrix produced for a structured suite |
| B | Gemma 4 multimodal task: confirm vision works through the adapter; add `multimodal` grader | One Gemma 4 image case passes; graded by the same json grader |
| C | Connect sequential gate to the runner loop (early stop, `inconclusive`) | A run stops early on a clear regression and reports the batch count |
| D | Bundle writer output in the report; `model-gate replay` command in coordination with A | Replay reproduces the same decision with zero model calls |

**Hours 30–38**

| Member | Tasks | Done when |
|--------|-------|-----------|
| A | Recorded-response mode for CI (read from `runs/recorded/`); cache-miss handling | Recorded run produces the same decision as the live run |
| B | Seeded regression: create the quantized candidate; document the expected failures | Failures reproduce consistently across three runs |
| C | Fix-hint templates for the top three failure classes; re-run only affected cases | One hint verified and one rejected in a demo run |
| D | Composite GitHub Action; PR comment with hidden marker; commit status mapping | Demo PR shows comment and status |

**Hours 38–44**

| Member | Tasks | Done when |
|--------|-------|-----------|
| A | Error handling, retries, `--max-spend` cap; clean exit codes | Timeouts and spend cap tested |
| B | Expand taxonomy coverage; review all example traces in the report for clarity | Every failure in the demo suite has a clear class |
| C | Policy edge cases (inconclusive handling, warn vs block); decision unit tests | Test suite covers all decision states |
| D | Demo repository setup; live run rehearsal; screenshots for the README | Full demo runs end to end twice |

**Hours 44–48**

| Member | Tasks |
|--------|-------|
| All | Bug fixes only, no new features after Hour 44 |
| D | Final README, docs/ci-recorded-mode.md, architecture diagram check |
| B | Sample suite documentation and privacy notes |
| C | Stats method explanation for judges (one page) |
| A | Install test on a clean machine; pip package check |
| All | Pitch rehearsal, Q&A prep, and demo backup video |

## 4. Integration Checkpoints

| Checkpoint | Time | What must work |
|------------|------|----------------|
| CP1 | Hour 6 | Adapter returns a Response; suite loads; a report renders from fake data |
| CP2 | Hour 14 | Real 10-case run produces a graded Markdown report |
| CP3 | Hour 22 | Policy blocks a seeded regression; Pareto view exists |
| CP4 | Hour 30 | Gemma 4 image case runs; constrained diagnosis works; early stop works |
| CP5 | Hour 38 | Replay works; GitHub Action comments on a demo PR |
| CP6 | Hour 44 | Full demo passes twice; feature freeze |

If a checkpoint slips by more than 4 hours, cut scope in this order: fix-hint loop (keep the template-free version), constrained decoding (keep free decoding only), HTML Pareto chart (keep table), OpenAI-compatible adapter (keep Ollama only). Do not cut the policy gate, the seeded regression demo, or the replay command.

## 5. Daily Sync

- Hour 0, Hour 12, Hour 24, Hour 36, Hour 44: 10-minute standup covering done, blocked, and next.
- Blocked items go to the owner of the blocking folder within 15 minutes.
- Demo script and README are owned by D, reviewed by everyone at Hour 40.

## 6. Smaller Teams

**Three members:** merge B and D. The member also writes the demo repository and the seeded regression. Keep the same hour plan but drop the HTML Pareto chart to a table if needed.

**Two members:** A and C together own the core (runner, adapters, stats, policy). B and D together own tasks, graders, reports, and the action. Cut the fix-hint loop and the OpenAI-compatible adapter first.

## 7. Definition of Done (Hackathon)

- Clone, install, and run the demo in under 5 minutes on a fresh machine
- Gate blocks the seeded regression and passes the known-good candidate
- Replay reproduces the decision with no model calls
- PR comment and commit status appear on the demo repository
- README, PRD, architecture, and flow docs match the implemented behavior
- Every claim in the pitch comes from a run that is saved in the repository
