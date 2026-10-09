"""Registry-backed runtime: resolve roles, report status, load the current model."""

from __future__ import annotations

from pathlib import Path

import yaml

from modelswap.results import DEFAULT_MAX_LATENCY_MS, EngineError, RoleStatus

DEFAULT_REGISTRY = Path("models.yaml")
CLI_DEFAULT_STATE = Path(".modelswap") / "state.json"
ROLE_ALIASES = {"classifier_role": "classifier"}


def resolve_registry_path(state_file: Path | None) -> Path:
    """Map the CLI state file to a registry path (the CLI default means models.yaml)."""
    if state_file is None or Path(state_file) == CLI_DEFAULT_STATE:
        return DEFAULT_REGISTRY
    return Path(state_file)


def load_registry(path: Path) -> dict:
    """Load the registry YAML; EngineError if missing, empty or without roles."""
    path = Path(path)
    if not path.is_file():
        raise EngineError(f"registry file not found: {path}")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not config.get("roles"):
        raise EngineError(f"registry {path} is empty or has no 'roles' section")
    return config


def resolve_role(config: dict, role: str) -> str:
    """Return the canonical role key, honouring aliases."""
    roles = config.get("roles") or {}
    name = ROLE_ALIASES.get(role, role)
    if name not in roles:
        valid = sorted(set(roles) | set(ROLE_ALIASES))
        raise EngineError(f"unknown role '{role}'; valid roles: {', '.join(valid)}")
    return name


def get_status(path: Path, role: str) -> RoleStatus:
    """Current and previous model for a role; previous is None when absent or equal to current."""
    config = load_registry(path)
    entry = config["roles"][resolve_role(config, role)]
    current = entry.get("current")
    previous = entry.get("previous")
    if not previous or previous == current:
        previous = None
    return RoleStatus(role=role, current=current, previous=previous)


def get_max_latency(config: dict, role: str) -> float:
    """Latency budget from the role thresholds, else the default."""
    entry = config["roles"][resolve_role(config, role)]
    value = (entry.get("thresholds") or {}).get("max_latency_ms")
    return float(value) if value is not None else DEFAULT_MAX_LATENCY_MS


def get(role: str, registry_path: Path | None = None):
    """Load the role's current model in eval mode."""
    from modelswap import arch

    path = resolve_registry_path(registry_path)
    status = get_status(path, role)
    model = arch.load_checkpoint(status.current)
    model.eval()
    return model
