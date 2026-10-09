"""Plain data types shared by every engine backend, the CLI and the demo.

Engines return these objects; the CLI and demo only render them. The default
thresholds mirror models.yaml, which is the source of truth when a role sets
its own values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_LATENCY_MS = 5.0  # models.yaml: thresholds.max_latency_ms
DEFAULT_MIN_ACCURACY = 0.85  # models.yaml: thresholds.min_accuracy
TARGET_SWAP_MS = 50.0  # PRD success criteria: swap and rollback under 50 ms


class EngineError(Exception):
    """Raised by an engine when an operation cannot run at all (bad input, no previous model)."""


@dataclass(frozen=True)
class ModelMetrics:
    model: str  # path of the model file
    accuracy: float  # 0.0 - 1.0
    latency_ms: float  # p95 single-request latency


@dataclass(frozen=True)
class CompareResult:
    role: str
    model_a: ModelMetrics  # current (champion)
    model_b: ModelMetrics  # candidate (challenger)
    eligible: bool  # the verdict: True means PASS
    reasons: tuple[str, ...] = field(default_factory=tuple)
    max_latency_ms: float = DEFAULT_MAX_LATENCY_MS

    @property
    def delta_accuracy(self) -> float:
        return self.model_b.accuracy - self.model_a.accuracy

    @property
    def delta_latency_ms(self) -> float:
        return self.model_b.latency_ms - self.model_a.latency_ms


def eligibility(
    model_a: ModelMetrics,
    model_b: ModelMetrics,
    max_latency_ms: float = DEFAULT_MAX_LATENCY_MS,
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
) -> tuple[bool, tuple[str, ...]]:
    """Same rule as modelswap.compare.verdict: no accuracy regression, minimum accuracy, latency budget."""
    reasons = []
    passed = True
    if model_b.accuracy < model_a.accuracy:
        passed = False
        reasons.append(f"candidate accuracy ({model_b.accuracy:.2%}) regressed below current ({model_a.accuracy:.2%})")
    if model_b.accuracy < min_accuracy:
        passed = False
        reasons.append(f"candidate accuracy ({model_b.accuracy:.2%}) is below minimum ({min_accuracy:.2%})")
    if model_b.latency_ms > max_latency_ms:
        passed = False
        reasons.append(f"candidate p95 latency ({model_b.latency_ms:.4f} ms) exceeds budget ({max_latency_ms:.4f} ms)")
    return passed, tuple(reasons)


@dataclass(frozen=True)
class WeightInfo:
    path: str
    params: int  # total number of scalar parameters
    tensors: int  # number of named tensors
    size_mb: float  # file size on disk
    l2_norm: float  # Frobenius norm over all parameters
    sha256: str  # hash of the tensor bytes (16 hex chars), not of the file


@dataclass(frozen=True)
class WeightReport:
    previous: WeightInfo | None  # the model that was active before the call
    new: WeightInfo | None  # the model the registry points to after a successful call
    delta_l2: float | None  # ||W_new - W_old||_F; None when architectures differ
    changed_tensors: int | None  # tensors whose values differ; None when architectures differ
    conversion_verified: bool | None  # True when the active model loads the new weights (None if nothing changed)
    note: str = ""


@dataclass(frozen=True)
class SwapResult:
    operation: str  # "apply" or "rollback"
    role: str
    ok: bool
    current: str  # active model after the call (unchanged when ok is False)
    previous: str | None
    smoke_test_passed: bool
    error: str | None = None
    weights: WeightReport | None = None


@dataclass(frozen=True)
class RoleStatus:
    role: str
    current: str
    previous: str | None
