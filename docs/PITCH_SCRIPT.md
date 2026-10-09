# 3-Minute Pitch Script

Owner: Anirudh. Run order: slides and live demo, in this order.

Slide-deck outline (7 slides). Build it in PowerPoint or Google Slides with the content below.

| # | Slide | Content | Time |
|---|---|---|---|
| 1 | Title | ML Model Swap: safe, reversible model updates. Team: Arjun, Bhagat, Achyut, Anirudh | 0:10 |
| 2 | Problem | Redeploying retrained models is manual and risky: no side-by-side check, no safe swap, no fast rollback | 0:25 |
| 3 | Solution | Apps ask for a role (`get("classifier_role")`), not a file path. Models are swapped by changing a pointer in `models.yaml` | 0:25 |
| 4 | Live demo | (switch to terminal) | 1:15 |
| 5 | Visual proof | MATLAB: decision boundary shift and the stream across the swap | 0:30 |
| 6 | Guarantees | Smoke-tested apply, one-command rollback, swap under 50 ms target | 0:15 |
| 7 | Ask / next | Open-source release; plug in any same-architecture MLP | 0:10 |

## Script

**Slide 1 (0:10)**
"We're the ML Model Swap team: Arjun, Bhagat, Achyut and me. This is a tool for swapping one machine learning model for another safely."

**Slide 2 (0:25)**
"Teams retrain models all the time. Today they often overwrite file paths by hand and redeploy blind. There's no check that the new model is actually better, and no quick way back if it's worse."

**Slide 3 (0:25)**
"Our app asks for a role, not a file. The swap engine compares two models with identical architecture, flips a pointer in one config file, runs a smoke test, and keeps the old model ready for rollback."

**Slide 4: live demo (1:15)**
Switch to the terminal and run `python -m modelswap demo` (with `--show-failure` if time allows).
- "We start on Model A, at 82% accuracy." (step 1–2)
- "Now compare: Model B reaches 94%, same input and output shapes, within our latency budget. Verdict: PASS." (step 3)
- "We try a broken candidate. The smoke test rejects it, and the active model stays A." (step 4, optional)
- "Apply promotes Model B. Watch the time: it's the pointer switch and the smoke test, well under 50 milliseconds." (step 5)
- Switch to MATLAB when the demo asks.

**Slide 5: MATLAB (0:30)**
"On the left, Model A's boundary; on the right, Model B's. The heatmap is the weight shift, and the stream line shows predictions continuing across the swap at t = 50. No gap."
Switch back, press Enter.

**Back in the terminal**
"Something looks wrong in production? One command." Run the rollback (step 6). "Back on Model A."

**Slide 6 (0:15)**
"Three guarantees: the candidate is smoke-tested before it goes live, a failed test leaves the old model in place, and rollback is one command."

**Slide 7 (0:10)**
"It's open source, and any two models with the same architecture work. Thanks. We're happy to take questions."

## Timing notes

- Total target: about 3:00. The live demo is the longest part; cut the optional failure step if you're over.
- Rehearse once with `python -m modelswap demo --fast` to check the timing, then once at full pace.
- If the demo breaks on stage, switch to the backup video (see `docs/DEMO_RUNBOOK.md`, section 6).
- Don't claim numbers the demo didn't show. If the real engine isn't wired in, say that the demo shows the stub's illustrative numbers.
