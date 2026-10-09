"""Real engine: registry-backed status, compare, apply, rollback and reset.

A role's `kind` in models.yaml picks the model family:
- "mlp" (default): the 2D MLP pair (Model A baseline vs Model B candidate). Identical architecture is required.
- "text_classification": fine-tuned BERT / RoBERTa. Each model is in its own class, so the
  role sets cross_architecture: true.

compare uses the verdict from the comparison code, so the CLI and the report always agree.
apply and rollback report the weights before and after, and check after the write that the
active model loads the new weights. If that check fails, models.yaml is restored byte for byte.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from modelswap import runtime, swap, weights
from modelswap.results import CompareResult, EngineError, ModelMetrics, RoleStatus, SwapResult

DEFAULT_AUDIT_DIR = Path("swaps")
TEXT_KIND = "text_classification"


def verify_candidate_architecture(candidate: str, current: str, entry: dict) -> tuple[bool, str]:
    """MLP pre-check: the candidate must have the same architecture as the current model and the registry entry."""
    from modelswap.compare import verify_architecture

    ok, mismatches = verify_architecture(current, candidate, entry)
    if ok:
        return True, "architecture matches"
    return False, "architecture mismatch: " + "; ".join(mismatches)


def _kind(entry: dict) -> str:
    return entry.get("kind", "mlp")


class RealEngine:
    """Implements the SwapEngine contract (see modelswap.engine) on top of models.yaml."""

    def __init__(self, registry_path: Path, audit_dir: Path = DEFAULT_AUDIT_DIR,
                 smoke_test=None, verify=None):
        self.registry_path = Path(registry_path)
        self.audit_dir = Path(audit_dir)
        self.smoke_test = smoke_test  # None: chosen per model kind
        self.verify = verify  # None: chosen per model kind

    # -- per-kind checks -------------------------------------------------------

    def _smoke_for(self, entry: dict):
        if self.smoke_test is not None:
            return self.smoke_test
        if _kind(entry) == TEXT_KIND:
            from modelswap.text import smoke_test_text

            return smoke_test_text
        return None  # swap.py default: MLP smoke test

    def _verify_for(self, entry: dict):
        if self.verify is not None:
            return self.verify
        if _kind(entry) == TEXT_KIND:
            from modelswap.text import verify_text_pair

            return verify_text_pair
        return verify_candidate_architecture

    # -- contract --------------------------------------------------------------

    def status(self, role: str) -> RoleStatus:
        return runtime.get_status(self.registry_path, role)

    def compare(self, role: str, candidate: str, data: str | None = None) -> CompareResult:
        config = runtime.load_registry(self.registry_path)
        key = runtime.resolve_role(config, role)
        entry = config["roles"][key]
        current = entry["current"]

        try:
            if _kind(entry) == TEXT_KIND:
                from modelswap.text import compare_text

                report = compare_text(
                    current, candidate,
                    data_path=data or entry["data"],
                    thresholds=entry.get("thresholds") or {},
                )
            else:
                from modelswap import compare as compare_module

                report = compare_module.compare(
                    (current, candidate),
                    data_path=data or "data/test.npz",
                    models_yaml=self.registry_path,
                    registry_entry=entry,
                )
        except ModuleNotFoundError:
            raise  # reported by the command layer as "needs a package that is not installed"
        except Exception as exc:
            raise EngineError(f"compare failed: {exc}") from exc

        if not report.get("architecture_ok", False) or not report.get("current"):
            reasons = "; ".join(report.get("verdict_reasons", [])) or "architecture mismatch"
            raise EngineError(f"cannot compare {current} with {candidate}: {reasons}")

        return CompareResult(
            role=role,
            model_a=ModelMetrics(current, report["current"]["accuracy"], report["current"]["latency_p95_ms"]),
            model_b=ModelMetrics(candidate, report["candidate"]["accuracy"], report["candidate"]["latency_p95_ms"]),
            eligible=report.get("verdict") == "PASS",
            reasons=tuple(report.get("verdict_reasons", [])),
            max_latency_ms=runtime.get_max_latency(config, role),
        )

    def apply(self, role: str, candidate: str) -> SwapResult:
        original = self.registry_path.read_bytes()
        config = runtime.load_registry(self.registry_path)
        entry = config["roles"][runtime.resolve_role(config, role)]
        previous_model, previous_pointer = entry["current"], entry.get("previous")
        result = swap.apply_swap(
            self.registry_path, role, candidate,
            smoke_test=self._smoke_for(entry), audit_dir=self.audit_dir, verify=self._verify_for(entry),
        )
        return self._with_weights(role, result, previous_model, candidate, original, previous_pointer)

    def rollback(self, role: str) -> SwapResult:
        original = self.registry_path.read_bytes()
        config = runtime.load_registry(self.registry_path)
        entry = config["roles"][runtime.resolve_role(config, role)]
        previous_model, previous_pointer = entry["current"], entry.get("previous")
        result = swap.rollback_swap(
            self.registry_path, role, smoke_test=self._smoke_for(entry), audit_dir=self.audit_dir,
        )
        return self._with_weights(role, result, previous_model, previous_pointer, original, previous_pointer)

    def reset(self, role: str) -> RoleStatus:
        return swap.reset_registry(self.registry_path, role)

    # -- weights report --------------------------------------------------------

    def _with_weights(self, role: str, result: SwapResult, previous_model: str,
                      new_model: str | None, original: bytes, previous_pointer: str | None) -> SwapResult:
        """Attach the before/after weights and, after a successful write, check the active model loads them."""
        report = weights.describe_pair(previous_model, new_model)
        if not result.ok:
            return replace(result, weights=report)

        report = weights.verify_active(report, runtime.get_status(self.registry_path, role).current)
        if report.conversion_verified is True:
            return replace(result, weights=report)

        # The registry points somewhere that does not hold the new weights. Put the old file back.
        swap.write_bytes_atomic(self.registry_path, original)
        return replace(
            result,
            ok=False,
            current=previous_model,
            previous=previous_pointer,
            error="post-write check failed: the active model does not load the new weights; models.yaml restored",
            weights=report,
        )


def create_engine(registry_path: Path | None = None, audit_dir: Path | None = None) -> RealEngine:
    registry = Path(registry_path or runtime.DEFAULT_REGISTRY)
    # Audit records sit next to the registry they describe (swaps/ beside models.yaml).
    return RealEngine(registry, audit_dir=audit_dir or registry.parent / DEFAULT_AUDIT_DIR.name)
