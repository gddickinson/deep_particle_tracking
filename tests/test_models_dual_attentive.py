"""DualBranchParticleTracker and AttentiveParticleTracker: forward pass and one training step on a tiny batch."""
import pytest

torch = pytest.importorskip('torch')

from models.losses import CombinedLoss  # noqa: E402
from models.network import create_model  # noqa: E402
from training.data_loader import SimulatedParticleDataset  # noqa: E402

TINY = {'dual': dict(base_filters=4, depth=2, num_frames=3),
        'attentive': dict(base_filters=8, depth=2, num_frames=3, heads=2)}


def _batch(size=32):
    ds = SimulatedParticleDataset(num_samples=2, frame_size=(size, size), frame_sequence_length=3,
                                  particles_per_frame=(2, 4))
    items = [ds[i] for i in range(2)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0] if k != 'frame_indices'}


@pytest.mark.parametrize('kind', ['dual', 'attentive'])
def test_forward_pass(kind):
    model = create_model(kind, **TINY[kind]).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 1, 32, 32))
    p = out['positions']
    assert p.shape[0] == 2 and p.shape[-3] == 1
    assert torch.isfinite(p).all() and float(p.min()) >= 0 and float(p.max()) <= 1
    assert out['embeddings'].shape[0] == 2


@pytest.mark.parametrize('kind', ['dual', 'attentive'])
def test_one_training_step(kind):
    torch.manual_seed(0)
    model = create_model(kind, **TINY[kind])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch()
    loss = CombinedLoss()(model(batch['frames']), batch)['total']
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('kind', ['dual', 'attentive'])
@pytest.mark.parametrize('factor', [1, 2, 8])
def test_upsample_factor_is_honoured(kind, factor):
    model = create_model(kind, **TINY[kind], upsample_factor=factor).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 3, 1, 16, 16))['positions'].shape[-2:] == (16 * factor, 16 * factor)


def test_super_resolution_peaks_map_to_frame_pixels(tmp_path):
    import numpy as np
    from prediction.inference import ParticlePredictor
    cfg = dict(TINY['dual'], upsample_factor=8)
    path = tmp_path / 'dual.pth'
    torch.save({'model_state_dict': create_model('dual', **cfg).state_dict(), 'model_type': 'dual',
                'model_config': cfg}, path)
    pred = ParticlePredictor(str(path), threshold=0.5, nms_radius=2)
    m = np.zeros((128, 128))
    m[8 * 5 + 3, 8 * 9 + 7] = 0.9                       # sub-pixel inside frame pixel (5, 9)
    m[8 * 5 + 4, 8 * 9 + 6] = 0.8                       # within the scaled NMS radius: suppressed
    pos = pred._postprocess_detection(m, (16, 16))
    np.testing.assert_allclose(pos, [[5 + 3.5 / 8 - 0.5, 9 + 7.5 / 8 - 0.5]])
    out = pred.predict_sequence(np.random.default_rng(0).random((4, 16, 16)).astype(np.float32), link_particles=False)
    allpos = np.concatenate([p.reshape(-1, 2) for p in out['positions']])
    assert len(allpos) == 0 or (allpos.min() >= -0.5 and allpos.max() <= 15.5)
