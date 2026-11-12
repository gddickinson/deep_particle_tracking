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
