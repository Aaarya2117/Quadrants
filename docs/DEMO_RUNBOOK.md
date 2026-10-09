# Demo Runbook (CLI and Demo Orchestration)

Owner: Anirudh. Use this to run, rehearse and record the live demo.

## 1. Setup (once)

Python 3.11+ is required. From the project root:

```powershell
python -m pip install -r requirements.txt
python -m pytest            # should end with: N passed
```

`requirements.txt` lists the runtime packages (PyYAML, numpy, scipy, scikit-learn, torch) and pytest.

## 2. Commands

| Command | What it does | Exit code |
|---|---|---|
| `python -m modelswap status` | Shows the active and previous model | 0 |
| `python -m modelswap compare --candidate models/model_b.pt` | Model A vs B on the test set, with the verdict and its reasons | 0 |
| `python -m modelswap apply --candidate models/model_b.pt` | Runs compare first, refuses unless the verdict is PASS, then pre-check, smoke test and atomic write | 0 ok, 1 refused or rejected |
| `python -m modelswap rollback` | Restores the previous model | 0 ok, 1 failed |
| `python -m modelswap reset` | Restores Model A as active (manual rehearsal only) | 0 |
| `python -m modelswap demo` | Full scripted demo on a temporary copy, pauses for Enter | 0 pass, 1 fail |

Options:
- Global, before the command: `--backend {auto,stub,real}` (default `auto`), `--registry PATH` (default `models.yaml`), `--state-file PATH` (stub only).
- `--role` on every command (default `classifier`; `classifier_role` also works).
- `apply --force`: apply even if the verdict is FAIL. Use only to show the override.
- `demo`: `--fast` removes the pauses; `--show-failure` adds a rejected corrupt-candidate step before the real swap.

## 3. The live demo, step by step

Run `python -m modelswap demo` (or `python -m modelswap --backend real demo`). The steps are:

1. Interactive Console Menu: Select model pair (`[1]` classifier MLP, `[2]` sentiment BERT/RoBERTa, `[3]` custom files).
2. Show the active model (`status`).
3. Compare A vs B on the validation set: Model A 85.1%, Model B 93.8%, delta accuracy +8.7%, verdict PASS.
4. Swap: promote Model B. Output shows swap time (<10ms), smoke test PASS, and Weight divergence block (Frobenius norm, tensor delta).
5. Roll back to Model A in one command.
6. Confirm the active model.

**The demo does not change the real files.** It copies `models.yaml` to a temporary folder and runs everything there. So `models.yaml` and `swaps/` in the project stay as they are, and you don't need `reset` before each run. The real `models.yaml` is only changed by the manual `apply`, `rollback` and `reset` commands.

The stub backend (`--backend stub`) runs the same steps with fixed numbers. Its banner says so.

## 4. Rehearsal checklist

- [ ] `python -m pytest` passes (105+ tests).
- [ ] `python -m modelswap --backend real demo --auto --fast --show-failure` ends with `Result: PASS (7/7 steps)`.
- [ ] Interactive demo (`python -m modelswap --backend real demo`) tested with menu navigation.
- [ ] Terminal font is large, the window is clean.
- [ ] If you ran manual `apply` or `rollback` during setup, run `python -m modelswap reset` so the demo starts from the state you expect.

## 5. Recording the backup video

1. Start the screen recorder (Windows: `Win + Alt + R` in Xbox Game Bar, or OBS).
2. Run `python -m modelswap --backend real demo` and navigate through the interactive menu.
3. Stop the recording.

## 6. Fallbacks

| If this fails | Do this |
|---|---|
| `apply` is refused on stage | This is the designed behaviour: the verdict did not pass, so nothing changed. Show the reasons on screen. |
| Real engine breaks | Run `python -m modelswap --backend stub demo --auto --fast`. The stub produces the same flow with fixed numbers. |
| Everything fails | Play the backup video. |

## 7. Where things live

| Path | Purpose |
|---|---|
| `modelswap/cli.py` | Argument parsing, exit codes, demo isolation |
| `modelswap/commands.py` | Runs one operation and prints it, including the apply gate (shared by CLI and demo) |
| `modelswap/demo.py` | Demo step list and runner |
| `modelswap/render.py` | Console output (ASCII only) |
| `modelswap/engine.py` | `SwapEngine` contract and backend selection |
| `modelswap/results.py` | Result types and the default thresholds |
| `modelswap/real_engine.py` | Real engine: compare via Bhagat's verdict, architecture pre-check, swap |
| `modelswap/stub_engine.py` | Stub engine with fixed numbers and a JSON state file |
| `tests/` | Tests (run with `python -m pytest`) |
| `swaps/` | Audit records from real `apply` and `rollback` (not created by the demo) |

## 8. Text role: BERT vs RoBERTa (sentiment)

The `sentiment` role compares `bert-base-uncased` (current) with `roberta-base` (candidate). Both are fine-tuned on SST-2 by the same script and settings. They are in different model classes, so the registry sets `cross_architecture: true` for this role.

Setup, once (downloads about 1 GB, and training takes roughly 10–20 minutes per model on CPU):

```powershell
python data/make_sst2.py
python -m models.text.train_text --base bert-base-uncased --out models/text/bert_base_uncased.pt --log-out models/text/bert_base_uncased_log.json
python -m models.text.train_text --base roberta-base --out models/text/roberta_base.pt --log-out models/text/roberta_base_log.json
```

The `.pt` checkpoints are large and are not committed (see `.gitignore`).

Compare, apply and roll back the text pair:

```powershell
python -m modelswap --backend real compare --role sentiment --candidate models/text/roberta_base.pt
python -m modelswap --backend real --registry models.yaml apply --role sentiment --candidate models/text/roberta_base.pt
python -m modelswap --backend real rollback --role sentiment
```

Every `apply` and `rollback` prints the **Weights (previous -> new)** block: the file of each model, its parameter count, its L2 norm and a tensor hash. For models with the same layout it also prints the delta `||W_new - W_old||_F`. After the write, the active model is reloaded from the registry, and the block says `conversion VERIFIED` only if its weights hash to the new model's hash. If that check fails, `models.yaml` is restored byte for byte.

Demo for this role (runs on a temporary copy, as with the MLP role):

```powershell
python -m modelswap --backend real demo --role sentiment --candidate models/text/roberta_base.pt --fast
```
