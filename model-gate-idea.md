# Model Gate: Evidence-Based Model Migration for Open-Weight AI

**Track:** PS 04, Build the AI System Behind the AI (Best Open-Source AI Project)

## Summary

Model Gate is an open-source evaluation and deployment-safety harness for swapping open-weight AI models. It treats a model change the way engineering teams already treat a code change: it runs a repeatable test suite, records evidence, and produces a clear merge recommendation before anything reaches production.

## The Problem

Teams usually choose a new model by looking at public leaderboard scores. Those scores say little about how a model will behave on a team's own prompts, tool schemas, output formats, latency targets, and cost limits. A migration that looks like an improvement on paper can quietly break production behavior, such as:

- Malformed function calls
- Invalid JSON output
- Accuracy drops on specific workflows
- Latency or cost overruns

Teams often discover these problems only after users complain.

## The Solution

Model Gate runs a team's own task suite against a baseline model and one or more candidate models, then reports the results with statistical rigor and a clear recommendation.

### Core Capabilities

1. **Pluggable task suites.** Tasks are defined in YAML or Python and can cover tool calling, structured extraction, JSON schema adherence, rubric-graded summarization, refusal and safety behavior, and multimodal tasks (Gemma 4's vision capabilities are a natural fit).
2. **Adapter layer.** A thin interface supports local servers (Ollama, vLLM, llama.cpp) and any OpenAI-compatible hosted endpoint. Adding a provider means writing one adapter class.
3. **Layered grading.** Deterministic graders run first (schema validation, exact match, regex, tool names, argument types). Only subjective cases go to an LLM-as-judge with a pinned judge model. Disputed judgments can be routed to a human review queue.
4. **Statistical comparison.** Each case runs several times at fixed seeds. Bootstrap confidence intervals and paired comparisons determine whether a difference is real, so regressions are flagged only when they are statistically significant.
5. **Cost and latency budgets.** Token usage, dollar cost, and p50/p95 latency are measured per task. A candidate that scores better but exceeds a budget can still be blocked.
6. **Failure taxonomy.** Failures are clustered into categories such as wrong tool name, missing required argument, invalid JSON, truncated output, and over-refusal, so the report explains what broke and why.
7. **Fix hints.** For recurring failure classes, Model Gate suggests prompt or schema changes, re-runs only the affected cases, and shows whether the fix holds.
8. **Reproducibility bundle.** Every run saves model identifiers, quantization settings, prompt hashes, seeds, and grader versions, so results can be reproduced and audited later.
9. **Gate policy file.** Blocking rules are written in plain language, for example: "block if tool-call accuracy drops by more than 2 points at 95% confidence, or if p95 latency grows more than 30%."

### Architecture

- **Runner:** async executor with per-provider rate limiting and caching keyed on (model, prompt hash, seed)
- **Adapter layer:** one class per backend
- **Grader layer:** deterministic graders first, LLM judge only when needed
- **Stats module:** bootstrap confidence intervals and paired comparisons
- **Report generator:** Markdown for PR comments, static HTML for review
- **Gate policy:** YAML rules that determine pass or block

### Delivery Surfaces

- Command-line tool
- GitHub Action that posts a Markdown report as a pull-request comment
- GitLab CI job
- Static HTML report

## Use Cases

- **Model selection:** compare any two open-weight models on your own tasks
- **Prompt regression testing:** rerun the harness when a prompt changes, not only when the model changes
- **Quantization and distillation checks:** confirm a smaller model still meets your quality bar before rollout
- **Compliance evidence:** the reproducibility bundle serves as an audit trail for model changes

## Hackathon Scope (48 Hours)

- CLI runner with the adapter interface and two backends (Ollama and one hosted OpenAI-compatible endpoint)
- Three task types: tool calls, JSON extraction, and one multimodal task using Gemma 4
- Deterministic graders with bootstrap confidence intervals
- GitHub Action that posts a PR comment
- Policy file that gates merges
- Template-based fix-hint step

## Demo Script (Under 3 Minutes)

1. Open a PR that changes `model: gemma-4-27b` to a smaller quantized variant.
2. The bot posts a report: tool-call accuracy down 4 points (significant), JSON validity unchanged, cost down 60%, p95 latency up 12%.
3. The gate blocks the merge and lists the failing case clusters with example traces.
4. The fix-hint step proposes a stricter schema prompt, re-runs the affected cases, and the gate turns green.

## Judging Angle

Lead with a measured example: a model change that looked like a 60% cost saving on the leaderboard reduced tool-call accuracy by four points on the team's real workload, and the gate caught it before merge. Judges respond to a concrete failure the tool prevented.

## Note on Claims

Report numbers in the demo should come from actual runs on the team's own suite. Avoid unsupported claims about the system's accuracy or generality.
