# Model Swap

**Compare, choose, and swap open-weight models in your codebase. Roll back in one command.**

Model Swap replaces an open-weight model in a running application without a hunt through the code. It converts downloaded weights into a format your backend can run, compares the model in production with a candidate on your own requests, shows the swap options and their risks, applies the chosen option by changing the model reference, runs a smoke test, and records the change. If anything is wrong, one command restores the previous model.

## Why

Switching models usually means editing hardcoded model names across the codebase, hoping the new model still produces the JSON and tool calls your code expects, and having no quick way back. Model Swap turns that into a reviewable, tested, reversible change.

## How It Works

Application code asks for a role, not a model:

```python
from modelswap import get

extractor = get("invoice_extractor")
result = extractor.chat(messages, tools=tools)
```

The role's model lives in `models.yaml`:

```yaml
roles:
  invoice_extractor:
    current:
      backend: ollama
      model: gemma3:12b
      quantization: q4_K_M
      digest: sha256:3f1a...
    previous:
      backend: ollama
      model: gemma3:12b
      quantization: q8_0
      digest: sha256:9c2b...
    params:
      temperature: 0
      max_tokens: 512
```

To swap, Model Swap changes `current` and moves the old value to `previous`. Application code does not change.

## Quick Start

Requirements: Python 3.11+, [Ollama](https://ollama.com) for local models.

```bash
pip install modelswap

# Convert downloaded weights into a candidate your backend can run
modelswap convert \
  --source hf:google/gemma-3-4b-it \
  --to gguf \
  --quantization q4_K_M \
  --out models/

# Compare the current model with a candidate on the role's recorded requests
modelswap compare \
  --role invoice_extractor \
  --candidate ollama:gemma3:4b-it-q4_K_M \
  --suite examples/suites/invoices.yaml \
  --out runs/demo

# See the swap options and their risk levels
modelswap options --run runs/demo

# Apply option 1 (shows the diff, asks for confirmation)
modelswap apply --run runs/demo --option 1

# Undo it
modelswap rollback --role invoice_extractor
```

Open `runs/demo/report.html` for the side-by-side comparison.

## Weight Conversion

Open-weight models are published in one format and run in another. `modelswap convert` turns downloaded weights into a model your backend can load, and registers the result as a swap candidate.

What it does:

1. **Verifies the source.** Checks the downloaded weights against a pinned digest, if one is given.
2. **Converts the format.** Tensors are renamed to the target's naming scheme and, where the target layout needs it, reshaped or transposed. The numbers themselves stay the same. The tokenizer and model settings (layers, context length) are copied into the file's metadata.
3. **Quantizes (optional).** Weights are stored with fewer bits, for example 16-bit to 4-bit, so the model is smaller and faster. This adds a small rounding error.
4. **Validates the output.** The tensor count and shapes match the source, the file loads, and a short test generation runs.
5. **Registers the result.** Computes the digest of the converted file and adds the model to the backend (for Ollama, this creates the model from the file).
6. **Records the conversion.** Writes a record to `conversions/` and prints a candidate reference to pass to `modelswap compare`.

| Source | Target | Backend |
|--------|--------|---------|
| Hugging Face (`safetensors`) | GGUF | Ollama, llama.cpp |

Conversion uses established converters (llama.cpp for GGUF) instead of reimplementing them. A converted model is never applied directly: quantization can change behavior, so it goes through the same compare, options, and apply path as any other candidate. Option 3 (Quantized variant) is produced by this command.

## Comparing Models

`modelswap compare` replays each recorded request through the current and candidate models and reports:

- **Contract:** does the output match what the calling code expects (tool names, required fields, JSON schema)?
- **Accuracy:** correct field values or correct tool calls, with a confidence interval on the difference
- **Format validity:** rate of outputs that parse and validate
- **Cost:** per 1,000 requests
- **Latency:** p50 and p95

Results are paired per request, so the difference between models is measured on the same inputs.

## Swap Options

`modelswap options` lists what you can do with the comparison:

| Option | What it does | Typical risk |
|--------|--------------|--------------|
| 1. Full swap | Switch the role to the candidate | Depends on metric differences |
| 2. Keep current | No change; shows why the candidate was not chosen | None |
| 3. Quantized variant | Use a smaller quantization of the candidate, if it passes | Lower if metrics hold |

Each option shows expected changes in accuracy, cost, and latency, the risk level from `policy.yaml`, and the exact steps. A candidate that fails the role's contract is listed only as "keep current," with the reasons.

## Applying and Rolling Back

`modelswap apply`:

1. Shows the diff of changes to `models.yaml` (and to source files, if the source path is used).
2. Asks for confirmation (`--yes` for CI).
3. Applies the change.
4. Runs a smoke test on the role's contract cases.
5. Rolls back automatically if the smoke test fails.
6. Writes an audit record to `swaps/`.

`modelswap rollback --role <role>` restores the previous reference, runs the same smoke test, and writes its own audit record.

## Legacy Code With Hardcoded Models

If your code has model names written directly into calls, run:

```bash
modelswap scan --path src/
```

The scanner lists each match with file, line, and confidence. Only high-confidence matches are eligible for automatic edits. Low-confidence matches are listed for manual change. Once the code uses `get("role")`, the registry path applies.

## Role Contracts

Each role has a contract in `contracts/<role>.yaml`:

```yaml
role: invoice_extractor
output:
  type: json
  schema:
    type: object
    properties:
      vendor: {type: string}
      total: {type: number}
      due_date: {type: string}
    required: [vendor, total, due_date]
```

A candidate must pass the contract on the suite before it can be applied.

## GitHub Action

On pull requests that change `models.yaml` or a candidate file, the action runs the comparison and posts the options as a comment. Applying a swap is a manual workflow dispatch with the chosen option, so no model change reaches the main branch without a person choosing it.

```yaml
on:
  pull_request:
    paths: ["models.yaml", "candidates/**"]
```

## Privacy

- Recorded requests are sanitized before they are stored. Model Swap scans for emails, phone numbers, and national ID patterns and masks matches.
- API keys are read from environment variables and never written to the registry, audit records, or reports.
- Do not commit suites that contain customer data.

## Limitations

- Passing the contract and the comparison lowers risk; it does not guarantee quality on inputs your suite does not cover.
- Prompts tuned for the old model may need adjustment for the new one. Model Swap reports this but does not rewrite prompts.
- Model weights are not stored in the repository. Converted files go to the `--out` directory (git-ignored). The registry records a digest, and the tool verifies it before applying.
- Conversion supports Hugging Face to GGUF only. It does not train or fine-tune weights, and a converted model can behave differently from the original, especially after quantization.
- Automatic staged or canary swaps are planned, not included in the hackathon build.

## Roadmap

- Staged swaps with automatic promotion and rollback on metric regression
- More conversion targets (ONNX, MLX) and more quantization schemes
- Integration with model registries for weight pulling and verification
- Hot reload in serving platforms, so applications pick up changes without restarting

## Contributing

Issues and pull requests are welcome. See `CONTRIBUTING.md` and the `CODE_OF_CONDUCT.md`.

## License

Apache-2.0. See `LICENSE`.
