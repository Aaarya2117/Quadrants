"""Real engine: registry-backed status, compare, apply, rollback and reset."""

from __future__ import annotations

from pathlib import Path

from modelswap import runtime, swap
from modelswap.results import CompareResult, EngineError, ModelMetrics, RoleStatus, SwapResult


class RealEngine:
    """Implements the SwapEngine contract (see modelswap.engine) on top of models.yaml."""

    def __init__(self, registry_path: Path, audit_dir: Path = Path("swaps"), smoke_test=None):
        self.registry_path = Path(registry_path)
        self.audit_dir = Path(audit_dir)
        self.smoke_test = smoke_test

    def status(self, role: str) -> RoleStatus:
        return runtime.get_status(self.registry_path, role)

    def compare(self, role: str, candidate: str, data: str | None = None) -> CompareResult:
        config = runtime.load_registry(self.registry_path)
        key = runtime.resolve_role(config, role)
        current = config["roles"][key]["current"]

        from modelswap import compare as compare_module

        try:
            report = compare_module.compare(
                (current, candidate),
                data_path=data or "data/test.npz",
                models_yaml=self.registry_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise EngineError(str(exc)) from exc
        if not report.get("architecture_ok", True) or not report.get("current"):
            reasons = "; ".join(report.get("verdict_reasons", [])) or "architecture mismatch"
            raise EngineError(f"cannot compare {current} with {candidate}: {reasons}")

        return CompareResult(
            role=role,
            model_a=ModelMetrics(current, report["current"]["accuracy"], report["current"]["latency_p95_ms"]),
            model_b=ModelMetrics(candidate, report["candidate"]["accuracy"], report["candidate"]["latency_p95_ms"]),
            max_latency_ms=runtime.get_max_latency(config, role),
        )

    def apply(self, role: str, candidate: str) -> SwapResult:
        return swap.apply_swap(
            self.registry_path, role, candidate, smoke_test=self.smoke_test, audit_dir=self.audit_dir
        )

    def rollback(self, role: str) -> SwapResult:
        return swap.rollback_swap(self.registry_path, role, smoke_test=self.smoke_test, audit_dir=self.audit_dir)

    def reset(self, role: str) -> RoleStatus:
        return swap.reset_registry(self.registry_path, role)


def create_engine(state_file: Path | None) -> RealEngine:
    return RealEngine(runtime.resolve_registry_path(state_file))
