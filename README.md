# Model Gate

**Replay your own traffic. Separate format failures from reasoning failures. Decide with statistics, not leaderboards.**

Model Gate is an open-source CI gate for swapping open-weight AI models. Point it at a baseline and a candidate model, give it a set of recorded requests from your own workload, and it returns a merge decision backed by paired statistics, a Pareto view of quality, cost, and latency, and a saved evidence bundle you can replay later.

## Why

Leaderboard scores do not tell you whether a model can produce the tool calls and JSON your application depends on. Model swaps, quantization, and prompt edits often fail quietly. Model Gate makes these changes testable the same way code changes are.

## Features

- **Trace replay:** run sanitized production-style requests offline against any model
- **Constraint-separated diagnosis:** tells you whether a failure is a format problem or a reasoning problem
- **Anytime-valid statistics:** blocks or passes early when evidence is clear, returns `inconclusive` when it is not
- **Pareto report:** quality vs cost vs p95 latency, with a policy that enforces your budgets
- **Failure taxonomy:** every failure gets a class such as `tool_name_wrong` or `json_schema_violation`
- **Fix hints:** suggests a change for a recurring failure class and re-runs only the affected cases
- **Reproducibility bundles:** replay any past decision without calling a model
- **GitHub Action:** posts a single PR comment and sets a commit status

## Quick Start

Requirements: Python 3.11+, and [Ollama](https://ollama.com) for local models.

```bash
pip install model-gate

# Pull a baseline and a candidate
ollama pull gemma3:12b
ollama pull gemma3:4b-it-q4_K_M

# Run the bundled example suite
model-gate run \
  --suite examples/suites/tools.yaml \
  --baseline ollama:gemma3:12b \
  --candidate ollama:gemma3:4b-it-q4_K_M \
  --policy examples/gate.yaml \
  --out runs/demo
```

Open `runs/demo/report.html` to see the decision, the Pareto chart, and failure classes.

Replay the same decision with no model calls:

```bash
model-gate replay runs/demo/bundle.json
```

## Defining a Task Suite

```yaml
# suites/orders.yaml
suite: orders-support
cases:
  - id: refund-001
    kind: tool_call
    input:
      messages:
        - role: user
          content: "I want a refund for order 8842, it arrived broken."
    tools:
      - name: create_refund
        parameters:
          order_id: {type: string, required: true}
          reason: {type: string, required: true}
    expect:
      tool: create_refund
      args_required: [order_id, reason]

  - id: extract-001
    kind: json_extract
    input:
      messages:
        - role: user
          content: "Invoice from Acme Ltd, total 1,240.50 INR, due 2026-11-02."
    schema:
      type: object
      properties:
        vendor: {type: string}
        total: {type: number}
        due_date: {type: string}
      required: [vendor, total, due_date]
    expect:
      vendor: "Acme Ltd"
      total: 1240.50
      due_date: "2026-11-02"
```

Supported kinds: `tool_call`, `json_extract`, `multimodal_extract` (adds an `image` field).

## Gate Policy

```yaml
# gate.yaml
rules:
  - metric: tool_call_accuracy
    max_drop: 0.02
    confidence: 0.95
    action: block
  - metric: json_valid_rate
    max_drop: 0.01
    action: block
  - metric: p95_latency_ms
    max_increase_pct: 30
    action: block
  - metric: cost_per_1k
    max_increase_pct: 0
    action: warn
inconclusive: warn
```

## GitHub Action

```yaml
# .github/workflows/model-gate.yml
name: Model Gate
on:
  pull_request:
    paths: ["models/**", "prompts/**", "schemas/**"]

jobs:
  gate:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      statuses: write
    steps:
      - uses: actions/checkout@v4
      - uses: your-org/model-gate-action@v0
        with:
          suite: suites/orders.yaml
          policy: gate.yaml
          baseline: models/baseline.yaml
          candidate: models/candidate.yaml
        env:
          OPENAI_COMPAT_API_KEY: ${{ secrets.MODEL_API_KEY }}
```

Recorded responses can be used in CI to keep runs fast and free. See [docs/ci-recorded-mode.md](docs/ci-recorded-mode.md).

## How a Decision Is Made

1. Every case runs with K seeds (default 3) on both models in free decoding.
2. Structured cases also run under constrained decoding when the adapter supports it.
3. Graders classify each failure.
4. Paired outcomes update a sequential test after each batch.
5. The policy checks each rule and returns `pass`, `block`, or `inconclusive`.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [FLOW.md](FLOW.md) for details.

## Interpreting the Report

- **Effect size and interval:** the change in each metric, with its 95% interval
- **Failure classes:** counts by taxonomy class, with example traces
- **Diagnosis matrix:** free vs constrained outcomes per case, showing format-only failures
- **Pareto chart:** candidates on the quality/cost and quality/latency planes

## Adapters

| Backend | Status | Constrained decoding |
|---------|--------|----------------------|
| Ollama | Supported | JSON schema format |
| OpenAI-compatible endpoint | Supported | Depends on server |
| llama.cpp server | Planned | GBNF grammars |
| vLLM | Planned | Guided decoding |

Write a new adapter by implementing `generate()` and `capabilities()` in `adapters/base.py`.

## Privacy

- Sanitize traces before using them. Model Gate scans for emails, phone numbers, and national ID patterns and warns when it finds them.
- API keys are read from environment variables and are never written to bundles or logs.
- Do not commit trace files that contain customer data.

## Limitations

- Results are only as good as the suite. A suite that does not reflect production traffic will not predict production behavior.
- LLM-as-judge is not in the MVP, so open-ended quality is not measured.
- Constrained-decoding diagnosis depends on backend support.
- Statistical guarantees hold for the tested cases, not for unseen inputs.

## Roadmap

- Shadow mode against sampled live traffic
- Calibrated LLM judge with human-labeled audit set
- Automatic quantization sweeps for Pareto search

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
