# Demo Runbook (CLI and Demo Orchestration)

Owner: Anirudh. Use this to run, rehearse and record the live demo.

## 1. Setup (once)

Python 3.11+ is required. The package uses only the standard library, so there is nothing to install.

From the project root (`D:\Arjun\projects\ModelGate`):

```powershell
python -m unittest discover -s tests -t .    # should end with: OK
```

## 2. Commands

| Command | What it does | Exit code |
|---|---|---|
| `python -m modelswap status` | Shows the active and previous model | 0 |
| `python -m modelswap compare --candidate models/model_b.pt` | Model A vs B on accuracy and latency, with a PASS/FAIL verdict | 0 |
| `python -m modelswap apply --candidate models/model_b.pt` | Atomic promote plus smoke test | 0 ok, 1 rejected |
| `python -m modelswap rollback` | Restores the previous model | 0 ok, 1 failed |
| `python -m modelswap reset` | Restores Model A as active (rehearsal only) | 0 |
| `python -m modelswap demo` | Full scripted demo, pauses for Enter between steps | 0 pass, 1 fail |

Options:
- Global, before the command: `--backend {auto,stub,real}` (default `auto`) and `--state-file PATH`.
- `--role` on every command (default `classifier_role`).
- `demo`: `--fast` removes the pauses; `--show-failure` adds a rejected broken-candidate swap before the real swap.

## 3. The live demo, step by step

Run `python -m modelswap demo`. The steps are:

1. Initial state: Model A serving (reset).
2. Show the active model (`status`).
3. Compare A vs B: the table shows 82% vs 94% and the verdict PASS.
4. (Optional with `--show-failure`) Try a broken candidate: the smoke test rejects it and the active model stays A.
5. Swap: promote Model B. The output shows the swap time. The demo then prints the MATLAB instruction and waits. Switch to MATLAB, run `matlab/simulate_swap.m` (Achyut's file), then come back and press Enter.
6. Roll back to Model A.
7. Confirm the active model.

Each step is also a plain CLI command, so if the demo script misbehaves you can type the commands by hand.

## 4. Rehearsal checklist

- [ ] `python -m unittest discover -s tests -t .` passes.
- [ ] `python -m modelswap reset` before every run, so the demo starts on Model A.
- [ ] `python -m modelswap demo --fast --show-failure` ends with `Result: PASS (7/7 steps)`.
- [ ] Full live run (no `--fast`) done at least twice, timed.
- [ ] Terminal font is large, the window is clean, and the MATLAB window is already open on `simulate_swap.m`.
- [ ] Demo runs on the real engine once it is wired in (see `docs/INTEGRATION.md`). The banner must say `backend=real`, not `backend=stub`.

## 5. Recording the backup video

1. Run `python -m modelswap reset`.
2. Start the screen recorder (Windows: `Win + Alt + R` in Xbox Game Bar, or OBS).
3. Run `python -m modelswap demo` and press Enter at each pause. Pause briefly at the MATLAB step.
4. Stop the recording. Save it as `demo_backup.mp4` outside the repo, or in `docs/` if the team wants it versioned.

## 6. Fallbacks

| If this fails | Do this |
|---|---|
| MATLAB does not open | Show the saved PNG of the decision boundaries and the stream plot (ask Achyut for a screenshot). |
| `apply` is rejected on stage | This is the designed behavior: point out that the active model is unchanged. Run the demo again after `reset`. |
| Real engine breaks | Run `python -m modelswap --backend stub demo --fast`. The stub produces the same flow with illustrative numbers. Say so. |
| Demo state is wrong | `python -m modelswap reset`, then run again. |
| Everything fails | Play the backup video. |

## 7. Where things live

| Path | Purpose |
|---|---|
| `modelswap/cli.py` | Argument parsing, exit codes |
| `modelswap/commands.py` | Runs one operation and prints it (shared by CLI and demo) |
| `modelswap/demo.py` | Demo step list and runner |
| `modelswap/render.py` | Console output (ASCII only) |
| `modelswap/engine.py` | `SwapEngine` contract and backend selection |
| `modelswap/results.py` | Result types and the eligibility rule |
| `modelswap/stub_engine.py` | Stub engine with a JSON state file |
| `tests/` | Unit tests (stdlib `unittest`) |
| `.modelswap/state.json` | Stub state file, created on first write (default location) |
