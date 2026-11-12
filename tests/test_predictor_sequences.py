"""ParticlePredictor feeds sliding windows of the trained length to the network (ConvLSTM sees T > 1)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from prediction.inference import ParticlePredictor  # noqa: E402
from models.network import create_model  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path):
    cfg = {'base_filters': 4, 'depth': 2, 'num_frames': 3}
    model = create_model('simple', **cfg)
    path = tmp_path / 'best.pth'
    torch.save({'model_state_dict': model.state_dict(), 'model_type': 'simple', 'model_config': cfg}, path)
    return str(path)


def test_sequence_windows_reach_the_model(checkpoint):
    pred = ParticlePredictor(checkpoint, threshold=0.5)
    seen = []
    pred.model.register_forward_pre_hook(lambda m, args: seen.append(tuple(args[0].shape)))
    frames = np.random.default_rng(0).random((6, 32, 32)).astype(np.float32)
    out = pred.predict_sequence(frames)
    assert len(out['positions']) == 6
    assert [s[1] for s in seen] == [1, 2, 3, 3, 3, 3]          # time axis: growing, then the trained length 3
    assert all(s[0] == 1 and s[2] == 1 and s[3:] == (32, 32) for s in seen)


def test_window_ends_at_the_predicted_frame(checkpoint):
    pred = ParticlePredictor(checkpoint)
    frames = np.random.default_rng(1).random((5, 32, 32)).astype(np.float32)
    maps = pred.predict_sequence(frames, link_particles=False, return_probability_maps=True)['probability_maps']
    direct, _ = pred._run_model(frames[2:5])
    np.testing.assert_allclose(maps[4], direct, rtol=1e-5, atol=1e-6)
