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


def test_loss_compares_with_the_last_frame():
    # The network predicts the last frame of its input window (decoder skips and ConvLSTM state from that frame),
    # so the localisation loss must use the last target frame, not the first.
    torch = pytest.importorskip('torch')
    from models.losses import CombinedLoss
    target = torch.zeros(1, 3, 1, 8, 8)
    target[0, 0, 0, 2, 2] = 1.0
    target[0, 2, 0, 5, 6] = 1.0
    pred_last = target[:, 2:3].clamp(0.01, 0.99)
    pred_first = target[:, 0:1].clamp(0.01, 0.99)
    loss = CombinedLoss(lambda_track=0.0)
    assert loss({'positions': pred_last}, {'positions': target})['loc_loss'] < \
        loss({'positions': pred_first}, {'positions': target})['loc_loss']
