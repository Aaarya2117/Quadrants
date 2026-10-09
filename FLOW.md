# Model Swap: Flows

Each flow lists its trigger, steps, decision points, and outputs.

## Flow 1: Compare Current vs Candidate

**Trigger:** `modelswap compare --role <role> --candidate <ref> --suite <file>`

```
1. Load configuration
   ├── models.yaml → current reference and params for the role
   ├── contracts/<role>.yaml → contract
   ├── suite file → recorded cases
   └── policy.yaml → budgets and alpha
          │
2. Validate inputs
   ├── candidate reference exists and its digest matches (if pinned)
   ├── suite cases match the role's input format
   └── failure → stop with the specific error
          │
3. Replay each case against both references
   for each case:
     ├── current reference  → response (cache by ref, prompt hash, params)
     └── candidate reference → response (cache by ref, prompt hash, params)
          │
4. Contract check on each candidate response
   ├── all cases pass          → continue to metrics
   └── contract failure rate above threshold
         → candidate marked "keep current", reasons recorded, go to step 7
          │
5. Grade and measure (paired per case)
   ├── accuracy and format validity for each side
   ├── cost per 1,000 requests and p50/p95 latency
   └── paired differences with intervals (McNemar for binary, bootstrap otherwise)
          │
6. Compute risk per metric using policy.yaml
   ├── significant accuracy drop      → high risk
   ├── budget exceeded (cost/latency) → medium risk
   └── all within tolerance           → low risk
          │
7. Write comparison outputs
   ├── comparison.json (complete, machine-readable)
   ├── report.md
   └── report.html
```

## Flow 2: Options

**Trigger:** `modelswap options --run <run>`

```
1. Load comparison.json
2. Build options:
   ├── if contract passed:
   │     1. Full swap (risk from comparison)
   │     3. Quantized variant, only if a quantized reference exists in the comparison
   ├── always:
   │     2. Keep current (with the main reasons)
   └── if contract failed:
         only "Keep current", with the failing contract checks
3. For each option, list:
   ├── expected changes in accuracy, cost, latency
   ├── risk level
   └── steps the tool will perform
4. Print the list and the id needed for apply
```

## Flow 3: Apply a Swap (Registry Path)

**Trigger:** `modelswap apply --run <run> --option <id> [--yes]`

```
1. Load option and its change set
2. Preconditions
   ├── comparison complete and contract passed
   ├── change set hash matches the reviewed hash
   ├── working tree clean, or change set is the only diff
   └── failure at any point → stop, nothing written
3. Show the diff of models.yaml
   ├── no --yes → ask for confirmation; "no" stops here
   └── --yes → continue (CI mode)
4. Save a copy of models.yaml (rollback source)
5. Write the new models.yaml atomically
   ├── current  ← candidate reference
   └── previous ← old current reference
6. Smoke test on the role's contract cases using the new current
   ├── pass → go to step 7
   └── fail → restore the saved copy, record failure, exit with code 1
7. Write audit record swaps/<timestamp>-<role>.json
   ├── comparison id, option id, diff, approver, smoke test result
   └── exit 0
```

## Flow 4: Apply a Swap (Source Path)

**Trigger:** `modelswap apply --run <run> --option <id>` where the option's change set includes source edits.

```
1. Preconditions as in Flow 3, plus:
   ├── every source edit has confidence above threshold
   └── target lines still match the expected old text (drift guard)
2. Create a branch swap/<role>-<timestamp> in git (or stop if git is not available)
3. Apply source edits
   ├── replace model string literal with get("<role>") (migration mode), or
   └── replace model string literal with the candidate string (direct mode)
4. Update models.yaml as in Flow 3 when migration mode is used
5. Smoke test as in Flow 3
   ├── pass → commit on the branch, write audit record, print the branch name
   └── fail → revert the branch commit, record failure, exit 1
6. The diff is ready for review; the user merges the branch
```

## Flow 5: Rollback

**Trigger:** `modelswap rollback --role <role>`

```
1. Read models.yaml
   ├── previous reference exists → continue
   └── no previous reference → stop with a message
2. Show the diff: current ↔ previous (confirmation unless --yes)
3. Swap the references: current ← previous, previous ← current
4. Smoke test on the role's contract cases
   ├── pass → write audit record swaps/<timestamp>-rollback-<role>.json
   └── fail → restore the file from the saved copy and report that rollback could not be verified
5. Exit 0 on success
```

For source-path swaps, rollback reverts the branch commit that the swap created, after the same smoke test.

## Flow 6: Source Scan

**Trigger:** `modelswap scan --path src/`

```
1. Walk files under the path (respecting the configured include and exclude patterns)
2. Parse each Python or JavaScript file into a syntax tree
3. Find model references:
   ├── string literal assigned to a model parameter (model="...")
   ├── known client constructor calls with a model argument
   └── references already using get("role") → marked as migrated
4. Score each match:
   ├── high  → exact call pattern recognized
   ├── medium → literal looks like a model name but the call is unknown
   └── low   → string matches a model name in a comment or unrelated context
5. Write scan report with file, line, match, confidence, and suggested role
```

Only high-confidence matches are eligible for automatic edits. Medium and low matches appear in the report for a person to decide.

## Flow 7: Weight Verification

**Trigger:** before a comparison or apply that uses a reference with a digest.

```
1. Resolve the reference to a local model (Ollama tag or local path)
2. Compute or read the model digest
3. Compare with the digest in models.yaml or the candidate reference
   ├── match → continue
   └── mismatch or missing model → stop with the expected and actual digest
```

No swap is applied with an unverified digest.

## Flow 8: Reviewing a Swap (Team Reader's Flow)

```
1. Open the comparison report
   └── check the contract result first
2. Read the options table
   └── note the risk level and the steps
3. Check the metric differences
   ├── accuracy: is the interval below the policy's threshold?
   ├── cost and latency: within the budgets?
   └── failure classes: which cases failed, and do they matter for the product?
4. Choose an option, or keep current
5. Apply, then confirm the smoke test and the audit record
```

## Flow 9: Pull Request Review (GitHub Action)

**Trigger:** pull request that changes `models.yaml` or files under `candidates/`.

```
1. Run Flow 1 for each role whose reference changed
2. Run Flow 2 to build options
3. Post or update one PR comment with the contract result, the metric differences, and the options
4. Set a status: success if at least one option is "full swap" with low or medium risk, neutral otherwise
5. Apply is not run by the action. The team applies a chosen option with the manual workflow dispatch, which runs Flow 3 or Flow 4
```
