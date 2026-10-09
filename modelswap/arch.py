from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

REQUIRED_CHECKPOINT_KEYS = {
    "state_dict",
    "input_dim",
    "hidden_dim",
    "output_dim",
    "activation",
    "architecture",
}

REQUIRED_SIGNATURE_KEYS = {
    "architecture",
    "input_dim",
    "hidden_dim",
    "output_dim",
    "activation",
}

class MLP(nn.Module):
    """
    2-layer Multi-Layer Perceptron (MLP) binary classifier.
    Architecture: Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.activation = "relu"
        self.architecture = "mlp_2layer"

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = self.relu(self.fc1(x))
        return self.fc2(h)

def save_checkpoint(model: MLP, path: str | Path) -> None:
    """
    Saves checkpoint dictionary containing weights, dimensions, and metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "output_dim": model.output_dim,
        "activation": "relu",
        "architecture": "mlp_2layer",
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str | Path) -> MLP:
    """
    Rebuilds the MLP from stored dimensions and loads weights.
    Raises KeyError if required checkpoint keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint at {path}, got {type(checkpoint).__name__}")

    missing_keys = REQUIRED_CHECKPOINT_KEYS - set(checkpoint.keys())
    if missing_keys:
        raise KeyError(
            f"Missing required checkpoint key(s) in {path}: {sorted(missing_keys)}. "
            f"Expected keys: {sorted(REQUIRED_CHECKPOINT_KEYS)}"
        )

    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def architecture_signature(path: str | Path) -> dict[str, Any]:
    """
    Reads architecture signature (dimensions, activation, architecture)
    WITHOUT instantiating or rebuilding the PyTorch model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint at {path}, got {type(checkpoint).__name__}")

    missing_keys = REQUIRED_SIGNATURE_KEYS - set(checkpoint.keys())
    if missing_keys:
        raise KeyError(
            f"Missing required signature key(s) in {path}: {sorted(missing_keys)}. "
            f"Expected keys: {sorted(REQUIRED_SIGNATURE_KEYS)}"
        )

    return {
        "architecture": checkpoint["architecture"],
        "input_dim": checkpoint["input_dim"],
        "hidden_dim": checkpoint["hidden_dim"],
        "output_dim": checkpoint["output_dim"],
        "activation": checkpoint["activation"],
    }
