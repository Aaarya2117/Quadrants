"""Weight fingerprints, deltas and conversion checks for swap reports.

describe_pair() reads the previous and the new checkpoint and reports their
fingerprints and, when both have the same tensor layout, how far the weights
moved. verify_active() reloads the file the registry now points to and checks
that it holds the new weights. That check is what "the weights were converted"
means here.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import replace
from pathlib import Path

import torch

from modelswap.results import WeightInfo, WeightReport


def load_state(path: str | Path) -> dict:
    """Return the state_dict stored in a checkpoint (MLP or text)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"no state_dict in checkpoint {path}")
    return checkpoint["state_dict"]


def tensor_hash(state: dict) -> str:
    """Hash of names, shapes and values, in a fixed order, so equal weights give equal hashes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:16]


def info_from_state(path: str | Path, state: dict) -> WeightInfo:
    params = sum(t.numel() for t in state.values())
    sq_sum = sum(float(t.detach().float().pow(2).sum()) for t in state.values())
    return WeightInfo(
        path=str(path),
        params=int(params),
        tensors=len(state),
        size_mb=round(os.path.getsize(path) / 1e6, 2),
        l2_norm=math.sqrt(sq_sum),
        sha256=tensor_hash(state),
    )


def delta(prev_state: dict, new_state: dict) -> tuple[float | None, int | None]:
    """(||W_new - W_old||_F, number of tensors that changed), or (None, None) if the layouts differ."""
    if prev_state.keys() != new_state.keys():
        return None, None
    if any(prev_state[k].shape != new_state[k].shape for k in prev_state):
        return None, None
    sq_sum = 0.0
    changed = 0
    for name in prev_state:
        diff = new_state[name].detach().float() - prev_state[name].detach().float()
        value = float(diff.pow(2).sum())
        sq_sum += value
        if value > 0:
            changed += 1
    return math.sqrt(sq_sum), changed


def describe_pair(previous_path: str | None, new_path: str | None) -> WeightReport:
    """Fingerprint the previous and new checkpoints. A file that can't be read is reported, not raised."""
    notes = []
    prev_info = prev_state = new_info = new_state = None

    if previous_path:
        try:
            prev_state = load_state(previous_path)
            prev_info = info_from_state(previous_path, prev_state)
        except Exception as exc:
            notes.append(f"previous unreadable: {exc}")
    if new_path:
        try:
            new_state = load_state(new_path)
            new_info = info_from_state(new_path, new_state)
        except Exception as exc:
            notes.append(f"new unreadable: {exc}")

    delta_l2 = changed = None
    if prev_state is not None and new_state is not None:
        delta_l2, changed = delta(prev_state, new_state)
        if delta_l2 is None:
            notes.append("different architectures: no tensor-by-tensor delta")

    return WeightReport(
        previous=prev_info,
        new=new_info,
        delta_l2=delta_l2,
        changed_tensors=changed,
        conversion_verified=None,
        note="; ".join(notes),
    )


def verify_active(report: WeightReport, active_path: str | Path) -> WeightReport:
    """Reload the active file from the registry and check it holds the new weights."""
    if report.new is None:
        return replace(report, conversion_verified=False, note=_join(report.note, "new weights unreadable"))
    try:
        active_state = load_state(active_path)
        active_hash = tensor_hash(active_state)
    except Exception as exc:
        return replace(report, conversion_verified=False, note=_join(report.note, f"active unreadable: {exc}"))
    verified = active_hash == report.new.sha256
    note = report.note if verified else _join(report.note, f"active hash {active_hash} != new {report.new.sha256}")
    return replace(report, conversion_verified=verified, note=note)


def _join(first: str, second: str) -> str:
    return "; ".join(part for part in (first, second) if part)
