"""web/local_models.py with a real (untrained, tiny) checkpoint. Skipped when PyTorch is not importable."""
import pytest

from web.local_models import find_checkpoints, predict, torch_status

pytestmark = pytest.mark.skipif(not torch_status()["available"], reason="PyTorch not importable")


def test_predict_with_tiny_untrained_checkpoint(tmp_path):
    import torch
    from models.network import create_model
    cfg = {"base_filters": 4, "depth": 2, "num_frames": 1}
    model = create_model("simple", **cfg)
    (tmp_path / "checkpoints" / "run1").mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict(), "model_type": "simple", "model_config": cfg},
               tmp_path / "checkpoints" / "run1" / "best.pth")
    assert find_checkpoints(tmp_path)[0]["path"] == "checkpoints/run1/best.pth"
    r = predict("checkpoints/run1/best.pth", 1, {"width": 32, "height": 32, "num_frames": 3, "num_particles": 3},
                root=tmp_path)
    assert set(r["model"]) >= {"f1", "mota"} and "baseline" in r
    with pytest.raises(FileNotFoundError):
        predict("../outside.pth", 1, {}, root=tmp_path)
