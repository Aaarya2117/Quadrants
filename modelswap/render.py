"""Plain-text rendering for CLI and demo output.

ASCII only, so the output prints cleanly in the Windows console.
"""

from __future__ import annotations

from modelswap.results import TARGET_SWAP_MS, CompareResult, RoleStatus, SwapResult, WeightInfo, WeightReport


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
        ["A (current)", result.model_a.model, f"{result.model_a.accuracy:.1%}", f"{result.model_a.latency_ms:.4f}"],
        ["B (candidate)", result.model_b.model, f"{result.model_b.accuracy:.1%}", f"{result.model_b.latency_ms:.4f}"],
    ]
    verdict = "PASS - candidate is eligible for apply" if result.eligible else "FAIL - candidate is not eligible"
    lines = [
        f"Comparison for role '{result.role}' (test set)",
        "",
        _table(["Model", "Path", "Accuracy", "Latency p95 (ms)"], rows),
        "",
        f"Delta accuracy: {result.delta_accuracy * 100:+.1f} pts",
        f"Delta latency:  {result.delta_latency_ms:+.4f} ms (budget <= {result.max_latency_ms:.1f} ms)",
        f"Verdict: {verdict}",
    ]
    lines += [f"  * {reason}" for reason in result.reasons]
    return "\n".join(lines)


def render_swap(result: SwapResult, elapsed_ms: float) -> str:
    if not result.ok:
        lines = [
            f"[FAIL] {result.operation} rejected for role '{result.role}': {result.error}",
            f"Active model unchanged: {result.current}",
        ]
    else:
        lines = [
            f"[OK] {result.operation} for role '{result.role}'",
            f"Current:    {result.current}",
            f"Previous:   {result.previous or '-'}",
            f"Smoke test: {'PASS' if result.smoke_test_passed else 'FAIL'}",
            f"Time:       {elapsed_ms:.2f} ms (target < {TARGET_SWAP_MS:.0f} ms)",
        ]
        if elapsed_ms >= TARGET_SWAP_MS:
            lines.append("[WARN] exceeded the swap time target")
    if result.weights is not None:
        lines += [""] + render_weights(result.weights)
    return "\n".join(lines)


def render_weights(report: WeightReport) -> list[str]:
    """Previous and new weights, the delta where the layouts match, and whether conversion was verified."""
    lines = ["Weights (previous -> new):"]
    lines.append(_weight_line("previous", report.previous))
    lines.append(_weight_line("new", report.new))
    if report.delta_l2 is not None:
        lines.append(
            f"  delta     ||W_new - W_old||_F = {report.delta_l2:.4f}"
            f" ({report.changed_tensors} of {report.new.tensors} tensors changed)"
        )
    else:
        lines.append("  delta     n/a (different architectures: no tensor-by-tensor delta)")
    if report.conversion_verified is True:
        lines.append("  conversion VERIFIED: the active model loads the new weights (hashes match)")
    elif report.conversion_verified is False:
        lines.append(f"  conversion NOT VERIFIED: {report.note}")
    else:
        lines.append("  conversion not applied: the active model is unchanged")
        if report.note:
            lines.append(f"  note      {report.note}")
    return lines


def _weight_line(label: str, info: WeightInfo | None) -> str:
    if info is None:
        return f"  {label:<9} unreadable"
    return (
        f"  {label:<9} {info.path}  params {info.params:,}  tensors {info.tensors}"
        f"  size {info.size_mb:.1f} MB  l2 {info.l2_norm:.2f}  hash {info.sha256}"
    )


def render_error(command: str, message: object) -> str:
    return f"[FAIL] {command}: {message}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * width for width in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)
