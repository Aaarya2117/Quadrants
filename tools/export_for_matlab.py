"""
Export PyTorch Model Weights and Evaluation Data for MATLAB Simulation.

Exports:
- exports/weights_a.mat: W1 (h x d), b1 (h x 1), W2 (c x h), b2 (c x 1)
- exports/weights_b.mat: W1 (h x d), b1 (h x 1), W2 (c x h), b2 (c x 1)
- exports/data.mat: X (n x d), y (n x 1), pred_a (n x 1), pred_b (n x 1)

All matrices are stored exactly as PyTorch stores them (no transposes).
In MATLAB:
  h1 = max(0, X * W1' + b1');  % or with column vectors: h1 = max(0, W1 * x + b1)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import scipy.io

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelswap.arch import load_checkpoint

def export_model_weights(model_path: str | Path, output_mat_path: str | Path) -> dict[str, tuple]:
    model_path = Path(model_path)
    output_mat_path = Path(output_mat_path)
    output_mat_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(model_path)
    state = model.state_dict()

    # Extract weight matrices exactly as PyTorch stores them (no transposes)
    W1 = state["fc1.weight"].detach().cpu().numpy().astype(np.float64)        # (h, d) -> (16, 2)
    b1 = state["fc1.bias"].detach().cpu().numpy().reshape(-1, 1).astype(np.float64)  # (h, 1) -> (16, 1)
    W2 = state["fc2.weight"].detach().cpu().numpy().astype(np.float64)        # (c, h) -> (2, 16)
    b2 = state["fc2.bias"].detach().cpu().numpy().reshape(-1, 1).astype(np.float64)  # (c, 1) -> (2, 1)

    mat_payload = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "architecture": "mlp_2layer",
        "activation": "relu",
    }
    scipy.io.savemat(str(output_mat_path), mat_payload)

    return {
        "W1": W1.shape,
        "b1": b1.shape,
        "W2": W2.shape,
        "b2": b2.shape,
    }

def export_all_for_matlab(
    model_a_path: str | Path = "models/model_a.pt",
    model_b_path: str | Path = "models/model_b.pt",
    data_path: str | Path = "data/test.npz",
    output_dir: str | Path = "exports"
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Weights
    shapes_a = export_model_weights(model_a_path, output_dir / "weights_a.mat")
    shapes_b = export_model_weights(model_b_path, output_dir / "weights_b.mat")

    # 2. Export Test Data & Python Predictions
    data_path = Path(data_path)
    if data_path.suffix == ".npz":
        data = np.load(data_path)
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64).reshape(-1, 1)
    elif data_path.suffix == ".csv":
        raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
        X = raw[:, :2].astype(np.float64)
        y = raw[:, 2].astype(np.int64).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")

    model_a = load_checkpoint(model_a_path)
    model_b = load_checkpoint(model_b_path)

    with torch.no_grad():
        preds_a = model_a(torch.from_numpy(X.astype(np.float32))).argmax(dim=-1).numpy().reshape(-1, 1)
        preds_b = model_b(torch.from_numpy(X.astype(np.float32))).argmax(dim=-1).numpy().reshape(-1, 1)

    acc_a = float((preds_a.flatten() == y.flatten()).mean())
    acc_b = float((preds_b.flatten() == y.flatten()).mean())

    data_payload = {
        "X": X,
        "y": y,
        "pred_a": preds_a,
        "pred_b": preds_b,
        "acc_a_python": acc_a,
        "acc_b_python": acc_b,
        "is_standardized": True,
        "standardization_note": "X is ALREADY standardized using train split statistics.",
    }
    scipy.io.savemat(str(output_dir / "data.mat"), data_payload)

    print("==================================================================")
    print("             MATLAB EXPORT COMPLETED (exports/*.mat)              ")
    print("==================================================================")
    print(f"Model A Weights Exported : {output_dir}/weights_a.mat")
    print(f"  W1 shape (h x d)       : {shapes_a['W1']}")
    print(f"  b1 shape (h x 1)       : {shapes_a['b1']}")
    print(f"  W2 shape (c x h)       : {shapes_a['W2']}")
    print(f"  b2 shape (c x 1)       : {shapes_a['b2']}")
    print(f"Model B Weights Exported : {output_dir}/weights_b.mat")
    print(f"  W1 shape (h x d)       : {shapes_b['W1']}")
    print(f"  b1 shape (h x 1)       : {shapes_b['b1']}")
    print(f"  W2 shape (c x h)       : {shapes_b['W2']}")
    print(f"  b2 shape (c x 1)       : {shapes_b['b2']}")
    print("------------------------------------------------------------------")
    print(f"Test Data & Predictions  : {output_dir}/data.mat")
    print(f"  X shape (n x d)        : {X.shape}")
    print(f"  y shape (n x 1)        : {y.shape} (labels 0..1)")
    print(f"  pred_a shape (n x 1)   : {preds_a.shape}")
    print(f"  pred_b shape (n x 1)   : {preds_b.shape}")
    print("------------------------------------------------------------------")
    print(f"PYTHON ACCURACY BENCHMARK (for Achyut's MATLAB parity test):")
    print(f"  Model A Accuracy       : {acc_a * 100:.2f}%")
    print(f"  Model B Accuracy       : {acc_b * 100:.2f}%")
    print(f"  Accuracy Delta (B - A) : {(acc_b - acc_a) * 100:+.2f}%")
    print("------------------------------------------------------------------")
    print("STANDARDIZATION STATUS   : X IS ALREADY STANDARDIZED.")
    print("  (Achyut should NOT re-scale X in simulate_swap.m)")
    print("==================================================================")

    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "shapes_a": shapes_a,
        "shapes_b": shapes_b,
        "x_shape": X.shape,
        "y_shape": y.shape,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Export weights and test data for MATLAB simulation.")
    parser.add_argument("--model-a", default="models/model_a.pt", help="Path to Model A checkpoint")
    parser.add_argument("--model-b", default="models/model_b.pt", help="Path to Model B checkpoint")
    parser.add_argument("--data", default="data/test.npz", help="Path to test data")
    parser.add_argument("--output-dir", default="exports", help="Output directory for .mat files")
    args = parser.parse_args()

    export_all_for_matlab(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        data_path=args.data,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
