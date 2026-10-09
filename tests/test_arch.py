from pathlib import Path
import pytest
import torch
from modelswap.arch import MLP, save_checkpoint, load_checkpoint, architecture_signature

def test_save_then_load_identical_outputs(tmp_path: Path):
    torch.manual_seed(42)
    model = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    checkpoint_path = tmp_path / "model_test.pt"

    # Save checkpoint
    save_checkpoint(model, checkpoint_path)
    assert checkpoint_path.exists()

    # Read signature without building model
    sig = architecture_signature(checkpoint_path)
    assert sig == {
        "architecture": "mlp_2layer",
        "input_dim": 2,
        "hidden_dim": 16,
        "output_dim": 2,
        "activation": "relu",
    }

    # Load rebuilt model
    loaded_model = load_checkpoint(checkpoint_path)

    # Random batch evaluation
    batch = torch.randn(8, 2)
    with torch.no_grad():
        orig_out = model(batch)
        loaded_out = loaded_model(batch)

    assert torch.allclose(orig_out, loaded_out, atol=1e-6)

def test_load_checkpoint_missing_keys_raises(tmp_path: Path):
    bad_checkpoint_path = tmp_path / "bad_checkpoint.pt"
    torch.save({"state_dict": {}}, bad_checkpoint_path)

    with pytest.raises(KeyError, match="Missing required checkpoint key"):
        load_checkpoint(bad_checkpoint_path)

    with pytest.raises(KeyError, match="Missing required signature key"):
        architecture_signature(bad_checkpoint_path)
