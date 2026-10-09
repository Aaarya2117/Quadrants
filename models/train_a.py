"""
Train Baseline MLP (Model A).

Trains on a SMALL subset of the training split for a limited number of epochs
to produce a deliberately modest baseline (approx. 84-85% test accuracy).

TUNING KNOBS (if accuracy is too high or too low):
- Lower accuracy: decrease --subset-size (e.g., 200 or 150) or decrease --epochs (e.g., 10).
- Higher accuracy: increase --subset-size (e.g., 350 or 500) or increase --epochs (e.g., 25).
- Hidden layer capacity: adjust --hidden-dim (default 16).
"""

import sys
from pathlib import Path
import json
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from modelswap.arch import MLP, save_checkpoint

def train_model_a(
    train_data_path: str | Path = "data/train.npz",
    val_data_path: str | Path = "data/val.npz",
    test_data_path: str | Path = "data/test.npz",
    model_output_path: str | Path = "models/model_a.pt",
    log_output_path: str | Path = "models/train_a_log.json",
    subset_size: int = 250,
    epochs: int = 15,
    lr: float = 0.015,
    hidden_dim: int = 16,
    seed: int = 42
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data splits
    train_data = np.load(train_data_path)
    val_data = np.load(val_data_path)
    test_data = np.load(test_data_path)

    X_train_full = train_data["X"]
    y_train_full = train_data["y"]

    # Use first subset_size samples
    X_sub = torch.from_numpy(X_train_full[:subset_size].astype(np.float32))
    y_sub = torch.from_numpy(y_train_full[:subset_size].astype(np.int64))

    X_val = torch.from_numpy(val_data["X"].astype(np.float32))
    y_val = torch.from_numpy(val_data["y"].astype(np.int64))

    X_test = torch.from_numpy(test_data["X"].astype(np.float32))
    y_test = torch.from_numpy(test_data["y"].astype(np.int64))

    # Initialize model
    model = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []

    model.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_sub)
        loss = criterion(logits, y_sub)
        loss.backward()
        optimizer.step()

        history.append({
            "epoch": ep,
            "loss": float(loss.item())
        })

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=-1)
        val_acc = float((val_preds == y_val).float().mean().item())

        test_preds = model(X_test).argmax(dim=-1)
        test_acc = float((test_preds == y_test).float().mean().item())

    # Save checkpoint
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, model_output_path)

    # Save training log
    log_output_path = Path(log_output_path)
    log_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "model_a",
            "subset_size": subset_size,
            "epochs": epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "seed": seed,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "loss_history": history
        }, f, indent=2)

    print("==================================================================")
    print("                    MODEL A TRAINING SUMMARY                      ")
    print("==================================================================")
    print(f"Architecture    : MLP 2-layer (input=2, hidden={hidden_dim}, output=2)")
    print(f"Training subset : {subset_size} samples (out of {len(y_train_full)} total train)")
    print(f"Epochs & LR     : {epochs} epochs, lr={lr}, seed={seed}")
    print(f"Validation Acc  : {val_acc * 100:.2f}%")
    print(f"Test Acc (1200) : {test_acc * 100:.2f}%")
    print(f"Saved model to  : {model_output_path}")
    print(f"Saved log to    : {log_output_path}")
    print("==================================================================")

    return {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "model_path": str(model_output_path),
        "log_path": str(log_output_path)
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline Model A.")
    parser.add_argument("--train-data", default="data/train.npz", help="Path to train NPZ")
    parser.add_argument("--val-data", default="data/val.npz", help="Path to val NPZ")
    parser.add_argument("--test-data", default="data/test.npz", help="Path to test NPZ")
    parser.add_argument("--model-out", default="models/model_a.pt", help="Path to save model checkpoint")
    parser.add_argument("--log-out", default="models/train_a_log.json", help="Path to save train log")
    parser.add_argument("--subset-size", type=int, default=250, help="Subset size for training")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_model_a(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        model_output_path=args.model_out,
        log_output_path=args.log_out,
        subset_size=args.subset_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
