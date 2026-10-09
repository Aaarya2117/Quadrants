"""Plain data types shared by every engine backend, the CLI and the demo.

Engines return these objects; the CLI and demo only render them. The
eligibility rule lives here so the stub and the real engine apply the same
policy.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LATENCY_MS = 15.0  # README: metric_threshold.max_latency_ms
TARGET_SWAP_MS = 50.0  # PRD success criteria: swap and rollback under 50 ms


class EngineError(Exception):
    """Raised by an engine when an operation cannot run at all (bad input, no previous model)."""


@dataclass(frozen=True)
class ModelMetrics:
    model: str  # path of the model file
    accuracy: float  # 0.0 - 1.0
    latency_ms: float


@dataclass(frozen=True)
class CompareResult:
    role: str
    model_a: ModelMetrics  # current (champion)
    model_b: ModelMetrics  # candidate (challenger)
    max_latency_ms: float = DEFAULT_MAX_LATENCY_MS

    @property
    def delta_accuracy(self) -> float:
        return self.model_b.accuracy - self.model_a.accuracy

    @property
    def delta_latency_ms(self) -> float:
        return self.model_b.latency_ms - self.model_a.latency_ms

    @property
    def eligible(self) -> bool:
        """PASS if the candidate is at least as accurate and within the latency budget (FLOW.md, Flow 1)."""
        return self.delta_accuracy >= 0 and self.model_b.latency_ms <= self.max_latency_ms


@dataclass(frozen=True)
class SwapResult:
    operation: str  # "apply" or "rollback"
    role: str
    ok: bool
    current: str  # active model after the call (unchanged when ok is False)
    previous: str | None
    smoke_test_passed: bool
    error: str | None = None


@dataclass(frozen=True)
class RoleStatus:
    role: str
    current: str
    previous: str | None
