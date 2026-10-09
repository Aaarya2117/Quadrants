# Model Gate: Flows

This document describes how Model Gate behaves end to end. Each flow lists its trigger, steps, decision points, and outputs.

## Flow 1: Full Gate Run (CLI)

**Trigger:** `model-gate run --suite ... --baseline ... --candidate ... --policy ...`

```
1. Load and validate config
   ├── suite YAML        → Suite model (fail fast on schema errors)
   ├── policy YAML       → Rule list
   └── model configs     → Adapter instances
          │
2. Sanitize and hash
   ├── scan cases for PII patterns → warn or abort (per --pii-mode)
   └── compute suite_hash, policy_hash
          │
3. Plan expansions
   for each case:
     for each model in {baseline, candidate}:
       for seed in 1..K:
         add expansion (free decoding)
         if case.kind is structured and adapter supports constraints:
           add expansion (constrained decoding)
          │
4. Execute in batches (default batch = 10 cases)
   ├── check cache by expansion_key
   ├── on miss: call adapter under per-provider semaphore
   ├── store Response
   └── grade Response → Outcome (deterministic)
          │
5. Aggregate per case
   └── case-level success rate per model per decoding mode
          │
6. Sequential update (after each batch)
   ├── compute paired statistics for each policy rule
   ├── update always-valid bounds, spend alpha
   └── decision state per rule: block | pass | continue
          │
   ┌──────┴─────────────────────────────┐
   │ any rule = block (confirmed)?      │──yes──> stop early, go to step 7
   └──────┬─────────────────────────────┘
          │ no
   ┌──────┴─────────────────────────────┐
   │ all rules = pass (confirmed)?      │──yes──> stop early, go to step 7
   └──────┬─────────────────────────────┘
          │ no
   ┌──────┴─────────────────────────────┐
   │ case budget or spend cap reached?  │──yes──> mark unresolved as inconclusive, go to step 7
   └──────┬─────────────────────────────┘
          │ no
          └──> return to step 4 with next batch
          │
7. Compute outputs
   ├── Pareto frontier (quality vs cost, quality vs p95)
   ├── diagnosis matrix (free vs constrained)
   ├── failure class counts and example traces
   └── policy decision with reasons per rule
          │
8. Write artifacts
   ├── bundle.json (replayable)
   ├── report.md
   ├── report.html
   └── decision.json
          │
9. Exit code
   ├── 0 = pass
   ├── 1 = block
   └── 2 = inconclusive (configurable to 0 or 1 in policy)
```

## Flow 2: Replay From Bundle

**Trigger:** `model-gate replay runs/demo/bundle.json`

```
1. Load bundle
2. Verify hashes: suite_hash, policy_hash, grader versions
   └── mismatch → refuse, print which input changed
3. For each stored Response:
   ├── regrade with the stored grader version
   └── recompute Outcome
4. Recompute statistics and policy decision
5. Compare with stored decision
   ├── identical → print "reproduced" and exit 0
   └── different → print diff and exit 3
```

No model is called in this flow.

## Flow 3: Fix-Hint Loop

**Trigger:** the run finished with a block or warn, and the user runs `model-gate hint runs/demo` (or the HTML report's "Try fix" button calls the same command).

```
1. Select failure classes
   └── classes with count >= N (default 3) in candidate failures
2. For each class, look up the template in hints/templates.py
   Examples:
   ├── tool_name_wrong      → add an explicit allowed-tools list to the system prompt
   ├── json_invalid         → enable constrained JSON decoding for this case set
   ├── truncated            → raise max_tokens by 1.5x
   └── tool_args_missing    → add required-argument reminder to the tool description
3. Apply the hint to a copy of the affected cases (never the originals)
4. Re-run only affected cases on the candidate, with K seeds
5. Compare failure class counts before and after
   ├── significant reduction (paired test)  → hint verified
   └── no significant change              → hint rejected, next template
6. Write hint_report.md listing:
   ├── hint text
   ├── affected cases
   ├── before and after counts
   └── verified or rejected
```

The original suite and config are never modified. A verified hint is printed as a patch the team can review and apply.

## Flow 4: Pull Request Gate (GitHub Action)

**Trigger:** pull request touching `models/**`, `prompts/**`, or `schemas/**`.

```
1. Checkout PR head and base
2. Restore recorded responses from the cache for base and head configs
   ├── recorded mode (default): use stored responses where expansion keys match
   └── live mode (scheduled or manual): call endpoints for missing expansions
3. Run Flow 1 with recorded or live adapters
4. Post or update a single PR comment (matched by a hidden marker)
   └── contents: decision, effect sizes, failure classes, Pareto summary, link to HTML report
5. Set commit status
   ├── pass          → success
   ├── block         → failure
   └── inconclusive  → neutral (or failure if configured)
6. Upload bundle and HTML report as workflow artifacts
```

Edge cases:

- **Forked PRs:** the action runs in recorded mode only, because secrets are unavailable. The comment notes that live mode is disabled.
- **Cache miss in recorded mode:** the case is marked `not_recorded` and excluded from the decision, and the comment lists how many cases were excluded. If exclusions exceed a threshold (default 10%), the decision becomes `inconclusive`.

## Flow 5: Trace Import

**Trigger:** `model-gate import traces.jsonl --out suites/imported.yaml`

```
1. Read JSONL line by line
2. Map each record to a case:
   ├── messages, tools, expected output or validator → case fields
   └── records missing expected output → flagged, not imported
3. Sanitize:
   ├── detect emails, phone numbers, national ID patterns
   ├── mask matches with stable placeholders (e.g. <EMAIL_1>)
   └── count matches per pattern
4. Deduplicate by case_hash
5. Write suite YAML and a sanitization report
6. Print counts: imported, flagged, deduplicated, masked
```

The sanitization report is a required review step before the suite is committed.

## Flow 6: Multimodal Case Execution

**Trigger:** a case with `kind: multimodal_extract` runs.

```
1. Load image from the path in the case (must be inside the suite directory)
2. Resize to the configured max edge and encode as base64
3. Check adapter capability `images`
   └── not supported → outcome = skipped_unsupported, excluded and reported
4. Send messages with image content parts
5. Grade as json_extract against the expected fields
6. Record image hash in the response so replays use the same input
```

## Flow 7: Decision Interpretation (Reader's Flow)

How a team member reads the report:

```
1. Read the top-line decision (pass / block / inconclusive)
2. Read the policy reasons
   └── which rule drove the decision, with its effect size and interval
3. Check the Pareto view
   └── is the candidate dominated on any axis the team cares about?
4. Open failure classes
   └── which class grew, and which examples show it
5. Check the diagnosis matrix
   ├── many format-only failures  → try constrained decoding or a schema fix
   └── many semantic failures     → the model is weaker on this task; consider another candidate
6. Decide: merge, reject, or run fix hints and re-check
```

## State Summary

| Object | States |
|--------|--------|
| Case result | `passed`, `failed`, `skipped_unsupported`, `not_recorded`, `error` |
| Rule decision | `block`, `pass`, `continue`, `warn` |
| Run decision | `pass`, `block`, `inconclusive` |
| Hint | `proposed`, `verified`, `rejected` |
