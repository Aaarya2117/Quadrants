# 3-Minute Pitch Script

Owner: Anirudh. Run order: slides and live demo, in this order.

Slide-deck outline (7 slides). Build it in PowerPoint or Google Slides with the content below.

| # | Slide | Content | Time |
|---|---|---|---|
| 1 | Title | ML Model Swap: safe, reversible model updates. Team: Arjun, Bhagat, Achyut, Anirudh | 0:10 |
| 2 | Problem | Redeploying retrained models is manual and risky: no side-by-side check, no safe swap, no fast rollback | 0:25 |
| 3 | Solution | Apps ask for a role (`get("classifier_role")`), not a file path. Models are swapped by changing a pointer in `models.yaml` | 0:25 |
| 4 | Live demo | (switch to terminal) | 1:15 |
| 5 | Weight Audit | Weight divergence: Frobenius norm ||W_new - W_old||_F, tensor delta, hash verification | 0:30 |
| 6 | Guarantees | Smoke-tested apply, one-command rollback, swap under 10 ms | 0:15 |
| 7 | Ask / next | Open-source release; supports MLPs and Transformers (BERT vs RoBERTa) | 0:10 |

## Script

**Slide 1 (0:10)**
"We're the ML Model Swap team: Arjun, Bhagat, Achyut and Anirudh. This is a tool for swapping one machine learning model for another safely."

**Slide 2 (0:25)**
"Teams retrain models all the time. Today they often overwrite file paths by hand and redeploy blind. There's no check that the new model is actually better, and no quick way back if it's worse."

**Slide 3 (0:25)**
"Our app asks for a role, not a file. The swap engine compares two models with identical architecture, flips a pointer in one config file, runs a smoke test, and keeps the old model ready for rollback."

**Slide 4: live demo (1:15)**
Switch to the terminal and run `python -m modelswap demo`.
- "We start on Model A, at 85% accuracy." (step 1–2)
- "Now compare: Model B reaches 94%, same input and output shapes, within our latency budget. Verdict: PASS." (step 3)
- "We try a corrupt candidate. The checks refuse it, and the active model stays A." (optional failure check)
- "Apply runs the verdict again, checks the architecture, runs the smoke test, and only then switches the pointer. The switch itself takes under 10 milliseconds."

**Slide 5: Weight Divergence Audit (0:30)**
"Notice the weight divergence report: it computes the Frobenius norm of parameter change across all tensors and verifies SHA256 parameter hashes to guarantee the active model actually loaded the new weights into memory. Furthermore, we support NLP transformers like fine-tuned BERT vs RoBERTa."

**Back in the terminal**
"Something looks wrong in production? One command." Run the rollback. "Back on Model A in 8 milliseconds."

**Slide 6 (0:15)**
"Three guarantees: the candidate is smoke-tested before it goes live, a failed test leaves the old model in place, and rollback is one command."

**Slide 7 (0:10)**
"It's open source, and any two models with the same architecture work. Thanks. We're happy to take questions."

## Timing notes

- Total target: about 3:00. The live demo is the longest part; cut the optional failure step if you're over.
- Rehearse once with `python -m modelswap demo --fast` to check the timing, then once at full pace.
- If the demo breaks on stage, switch to the backup video (see `docs/DEMO_RUNBOOK.md`, section 6).
- Don't claim numbers the demo didn't show. If the real engine isn't wired in, say that the demo shows the stub's illustrative numbers.

## Questions to expect (have these ready)

- **"Isn't the baseline a strawman?"** Yes, on purpose. Model A was trained on 250 of the 3,600 training samples for 15 epochs (see `models/README.md`). The point of the demo is the swap, so the baseline is deliberately modest. Say so.
- **"Is 93.75% vs 85.08% significant?"** On the 1,200-sample test set the two models disagree on 130 points: B is right on 117 and A on 13. A McNemar test gives chi-squared of about 81.6 (1 degree of freedom), so p < 0.001.
- **"Is the new model faster?"** No claim. The p50 latency difference is about -0.0001 ms, which is noise. Say "no latency regression", not "faster".
- **"What does the rollback time include?"** The CLI starts a new process each time, so the first call loads PyTorch inside the timed step. The pointer switch itself is a few milliseconds. The 50 ms target is for the switch in a running service.
