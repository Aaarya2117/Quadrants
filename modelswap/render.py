"""Plain-text rendering for CLI and demo output.

ASCII only, so the output prints cleanly in the Windows console.
"""

from __future__ import annotations

from modelswap.results import TARGET_SWAP_MS, CompareResult, RoleStatus, SwapResult


def render_status(status: RoleStatus) -> str:
    return "\n".join(
        [
            f"Role:     {status.role}",
            f"Current:  {status.current}",
            f"Previous: {status.previous or '-'}",
        ]
    )


def render_compare(result: CompareResult) -> str:
    rows = [
        ["A (current)", result.model_a.model, f"{result.model_a.accuracy:.1%}", f"{result.model_a.latency_ms:.2f}"],
        ["B (candidate)", result.model_b.model, f"{result.model_b.accuracy:.1%}", f"{result.model_b.latency_ms:.2f}"],
    ]
    verdict = "PASS - candidate is eligible for apply" if result.eligible else "FAIL - candidate is not eligible"
    return "\n".join(
        [
            f"Comparison for role '{result.role}' (validation replay)",
            "",
            _table(["Model", "Path", "Accuracy", "Latency (ms)"], rows),
            "",
            f"Delta accuracy: {result.delta_accuracy * 100:+.1f} pts",
            f"Delta latency:  {result.delta_latency_ms:+.2f} ms (budget <= {result.max_latency_ms:.1f} ms)",
            f"Verdict: {verdict}",
        ]
    )


def render_swap(result: SwapResult, elapsed_ms: float) -> str:
    if not result.ok:
        return "\n".join(
            [
                f"[FAIL] {result.operation} rejected for role '{result.role}': {result.error}",
                f"Active model unchanged: {result.current}",
            ]
        )
    lines = [
        f"[OK] {result.operation} for role '{result.role}'",
        f"Current:    {result.current}",
        f"Previous:   {result.previous or '-'}",
        f"Smoke test: {'PASS' if result.smoke_test_passed else 'FAIL'}",
        f"Time:       {elapsed_ms:.2f} ms (target < {TARGET_SWAP_MS:.0f} ms)",
    ]
    if elapsed_ms >= TARGET_SWAP_MS:
        lines.append("[WARN] exceeded the swap time target")
    return "\n".join(lines)


def render_error(command: str, message: object) -> str:
    return f"[FAIL] {command}: {message}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * width for width in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)
