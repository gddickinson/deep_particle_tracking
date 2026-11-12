"""Training step: the simple model's output goes through CombinedLoss (FocalLoss expects probabilities)."""
import pytest


def test_simple_model_trains_with_the_default_loss():
    torch = pytest.importorskip('torch')
    from models.losses import CombinedLoss
    from models.network import create_model
    from training.data_loader import SimulatedParticleDataset
    ds = SimulatedParticleDataset(num_samples=2, frame_size=(32, 32), frame_sequence_length=3, particles_per_frame=(2, 4))
    batch = {k: v.unsqueeze(0) for k, v in ds[0].items()}
    model = create_model('simple', base_filters=4, depth=2, num_frames=3)
    losses = CombinedLoss()(model(batch['frames']), batch)
    losses['total'].backward()
    assert torch.isfinite(losses['total'])


def test_position_output_is_a_probability():
    torch = pytest.importorskip('torch')
    from models.network import create_model
    out = create_model('simple', base_filters=4, depth=2, num_frames=2)(torch.randn(1, 2, 1, 16, 16) * 5)
    p = out['positions']
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
