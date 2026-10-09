# Model Swap: Team Assignments

**Team:** Arjun, Bhagat, Achyut, Anirudh
**Reference:** the detailed hour plan is in `WORK_DIVISION.md`. This sheet covers ownership and handoffs for the revised scope.

## 1. Owners

| Member | Role | Owns | Main deliverables |
|--------|------|------|-------------------|
| **Arjun** | Runtime and Swap Engine | `runtime/`, `swap/`, `source/`, `cli.py` | `models.yaml` registry, `get(role)`, Ollama and OpenAI-compatible backends, registry apply and rollback, atomic writes, source scanner and editor, audit records |
| **Bhagat** | Suites and Comparison | `suite/`, `compare/`, `contracts/`, demo project | Recorded suite format and importer, PII masking, contract checker, async replay runner with caching, graders, demo project with three roles and a seeded bad candidate |
| **Achyut** | Weight Conversion, Statistics and Options | `convert/`, `options/`, `policy/` (in `compare/stats.py` with Bhagat's runner interface) | Weight conversion (Hugging Face to GGUF, quantization, validation, conversion records), weight digest verification, paired statistics (McNemar, bootstrap), cost and latency metrics, risk rules in `policy.yaml`, option builder |
| **Anirudh** | Reports, CI, and Demo | `report/`, `ci/action/`, demo repository, presentation | Markdown and HTML reports, options page, GitHub Action (comparison comment and manual apply dispatch), README and pitch, Gemma 4 multimodal role |

Basis for these roles: Arjun takes the swap engine because it is the most safety-critical part and needs the tightest control over file writes. Bhagat owns the data and contract side, since every decision depends on it. If a member's strengths point elsewhere, swap roles in the first hour and keep the folder ownership with whoever takes each role.

## 2. Ownership Rules

- Edit only the folders you own. For a change elsewhere, open a short pull request and tag the owner.
- `contracts.py` (shared types: `ModelRef`, `Response`, `Contract`, `ChangeSet`, `Option`) is agreed in Hours 0–1 by all four. After Hour 1, changes need all four to approve.
- Arjun reviews any change to `swap/`. Achyut reviews any change to risk logic in `options/` and `policy.yaml` and any change to `convert/`.
- `convert/` never writes `models.yaml`. Registry changes happen only in `swap/`.
- No one merges changes to `source/editor.py` without a test that checks the drift guard.

## 3. First Actions (Hours 0–1)

| Member | First task |
|--------|-----------|
| Arjun | Create the repository and license; write `models.yaml` schema and the `get(role)` loader |
| Bhagat | Write 15 seed cases for the invoice role and the `invoice_extractor` contract |
| Achyut | Write the risk rules and the `policy.yaml` format; define option types; get the llama.cpp converter running on one small Hugging Face model |
| Anirudh | Set up the report template and the PR comment layout; choose the Gemma 4 model tag |
| All | Agree the shared types in `contracts.py` before writing logic |

## 4. Handoffs

| From | To | Handoff | Needed by |
|------|----|---------|-----------|
| Arjun | Bhagat | Working `get(role)` returning a client that calls Ollama | Hour 6 |
| Achyut | Arjun | `convert` output: a candidate reference registered in Ollama, plus the conversion record format | Hour 8 |
| Achyut | Bhagat | Converted and quantized candidate models for the demo roles | Hour 14 |
| Bhagat | Achyut | Paired case results with contract and grading outcomes | Hour 14 |
| Achyut | Anirudh | Comparison summary with options and risk levels | Hour 22 |
| Arjun | Anirudh | Audit record format and apply output | Hour 30 |
| Arjun | Bhagat | Apply and rollback working on the demo project | Hour 30 |
| Bhagat | Arjun | Demo project with the seeded bad candidate for end-to-end tests | Hour 36 |

## 5. Checkpoints

| Checkpoint | Time | Owner who reports | What must work |
|------------|------|------------------|----------------|
| CP1 | Hour 6 | Arjun | `get("role")` calls the model; `models.yaml` loads |
| CP2 | Hour 14 | Bhagat and Achyut | `modelswap convert` yields a validated GGUF candidate; comparison on 15 cases produces paired results and a contract verdict |
| CP3 | Hour 22 | Achyut | Options list shows risk; bad candidate is "keep current" |
| CP4 | Hour 30 | Arjun | Apply on the registry path passes smoke test; rollback restores the file byte-for-byte |
| CP5 | Hour 38 | Arjun and Anirudh | Source path applies on the demo project; PR comment posts from the action |
| CP6 | Hour 44 | All | Full cycle on the demo passes twice; feature freeze |

If a checkpoint slips by more than 4 hours, cut in this order: staged option (already out of scope), source path (keep registry path only, and state it in the demo), HTML report (keep Markdown), the OpenAI-compatible backend (keep Ollama). Do not cut contract checks, the smoke test, rollback, the audit record, or the core conversion path (Hugging Face to GGUF, quantization, validation). Extra quantization schemes and other targets are cut first.

## 6. Daily Syncs

- Hours 0, 12, 24, 36, and 44: a 10-minute standup. Each member gives: done, blocked, next.
- Blocked items go to the owner of the blocking folder within 15 minutes.

## 7. Final-Day Responsibilities

| Member | Hours 44–48 |
|--------|-------------|
| Arjun | Install test on a clean machine; confirm the rollback test passes; tag the release |
| Bhagat | Document the suite format and the privacy masking; check the demo data is synthetic |
| Achyut | One-page explanation of the options and risk rules for judges, and one page on how conversion and quantization work and how they are validated |
| Anirudh | Finalize README, architecture diagram, demo backup video, and PR comment screenshots |
| All | Bug fixes only; pitch rehearsal; Q&A preparation |
