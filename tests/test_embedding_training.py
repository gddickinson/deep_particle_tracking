"""The association embeddings are trained: every model emits one embedding map per input frame, EmbeddingLoss pairs
them across frames by the simulator's ground-truth track IDs, and the predictor links with embeddings from the
same window. (Before: one frame of embeddings, so the loss had no frame pairs, was 0 and never trained them.)"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from models.losses import CombinedLoss  # noqa: E402
from models.network import create_model  # noqa: E402
from training.data_loader import SimulatedParticleDataset  # noqa: E402

T = 3
CONFIGS = {
    'simple': dict(base_filters=4, depth=2, num_frames=T, embedding_dim=8),
    'dual': dict(base_filters=4, depth=2, num_frames=T, upsample_factor=2),
    'attentive': dict(base_filters=8, depth=2, num_frames=T, heads=2, upsample_factor=2),
}


def _embedding_head(model):
    """The layers that produce the embeddings (Dual Branch: the association decoder's last block)."""
    return model.assoc_decoders[-1] if hasattr(model, 'assoc_decoders') else model.embedding_conv


def _batch(n=2, size=32, particles=(4, 8), start=0):
    ds = SimulatedParticleDataset(num_samples=start + n, frame_size=(size, size), frame_sequence_length=T,
                                  particles_per_frame=particles)
    items = [ds[start + i] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


@pytest.mark.parametrize('model_type', list(CONFIGS))
def test_embedding_loss_is_nonzero_and_reaches_the_embedding_head(model_type):
    torch.manual_seed(0)
    batch = _batch()
    model = create_model(model_type, **CONFIGS[model_type])
    out = model(batch['frames'])
    emb = out['embeddings']
    assert emb.dim() == 5 and emb.shape[:2] == (2, T), 'one embedding map per input frame'
    assert emb.shape[-2:] == batch['frames'].shape[-2:], 'embeddings at frame resolution'

    losses = CombinedLoss(track_warmup_steps=0)(out, batch)   # full tracking weight from the first step
    assert float(losses['track_loss'].detach()) > 0
    losses['total'].backward()
    grads = [p.grad for p in _embedding_head(model).parameters()]
    assert all(g is not None for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


@pytest.mark.parametrize('model_type', list(CONFIGS))
def test_every_model_still_trains_and_predicts_the_last_frame(model_type):
    torch.manual_seed(0)
    batch = _batch(n=1)
    model = create_model(model_type, **CONFIGS[model_type])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss = CombinedLoss()(model(batch['frames']), batch)['total']
    loss.backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        full = model(batch['frames'])
        # positions still describe the window's last frame (one map, not one per frame)
        assert full['positions'].reshape(-1).numel() == \
            CONFIGS[model_type].get('upsample_factor', 1) ** 2 * batch['frames'][0, 0].numel()


def _pair_distances(emb, track_ids):
    """Mean embedding distance between particles in different frames of each window: same ID vs different ID."""
    same, diff = [], []
    for b in range(emb.shape[0]):
        per_frame = []
        for f in range(emb.shape[1]):
            ids = track_ids[b, f, 0]
            ys, xs = torch.nonzero(ids > 0, as_tuple=True)
            per_frame.append((emb[b, f][:, ys, xs].T, ids[ys, xs]))
        for f1 in range(len(per_frame)):
            for f2 in range(f1 + 1, len(per_frame)):
                (e1, i1), (e2, i2) = per_frame[f1], per_frame[f2]
                d = torch.cdist(e1, e2)
                m = i1[:, None] == i2[None, :]
                same.append(d[m])
                diff.append(d[~m])
    return float(torch.cat(same).mean()), float(torch.cat(diff).mean())


def test_tiny_training_separates_same_and_different_tracks():
    # Seeded, CPU, ~15 s: 200 steps on fresh simulated windows (4 per step, 6-12 particles, 32 x 32 px), scored on
    # 16 held-out windows. Distances are measured with BatchNorm on batch statistics (train mode, no gradients) so
    # the untrained running statistics do not distort the baseline.
    torch.manual_seed(0)
    cfg = dict(CONFIGS['simple'], base_filters=8)
    model = create_model('simple', **cfg)
    ds = SimulatedParticleDataset(num_samples=10000, frame_size=(32, 32), frame_sequence_length=T,
                                  particles_per_frame=(6, 12))

    def batch(idx):
        items = [ds[i] for i in idx]
        return {k: torch.stack([it[k] for it in items]) for k in items[0]}

    held_out = batch(range(5000, 5016))
    loss_fn = CombinedLoss(lambda_track=1.0, track_warmup_steps=0)
    opt = torch.optim.Adam(model.parameters(), 1e-2)

    def gap():
        with torch.no_grad():
            same, diff = _pair_distances(model(held_out['frames'])['embeddings'], held_out['track_ids'])
        return (diff - same) / diff    # > 0: same-ID pairs are closer (more similar) than different-ID pairs

    before = gap()
    for step in range(200):
        b = batch(range(4 * step, 4 * step + 4))
        opt.zero_grad()
        loss_fn(model(b['frames']), b)['total'].backward()
        opt.step()
    after = gap()
    assert after > 0.35 and after > before + 0.25, (before, after)


def test_predictor_links_with_embeddings_from_the_same_window(tmp_path, monkeypatch):
    # Transition t-1 -> t compares frame t-1's and frame t's embeddings from the window ending at t (the pair the
    # loss trains), not embeddings taken from two different windows.
    from prediction import predictor as predictor_module
    from prediction.predictor import ParticlePredictor
    from prediction.preprocess import normalize_intensity
    cfg = CONFIGS['simple']
    model = create_model('simple', **cfg)
    path = tmp_path / 'best.pth'
    torch.save({'model_state_dict': model.state_dict(), 'model_type': 'simple', 'model_config': cfg}, path)
    pred = ParticlePredictor(str(path), threshold=-1e9, nms_radius=4)
    seen = {}
    real = predictor_module.link_particles_with_embeddings

    def spy(p, d, m, *a, **kw):
        seen['prev'] = kw.get('prev_descriptors')
        return real(p, d, m, *a, **kw)
    monkeypatch.setattr(predictor_module, 'link_particles_with_embeddings', spy)
    frames = np.random.default_rng(1).random((4, 32, 32)).astype(np.float32)
    res = pred.predict_sequence(frames, max_distance=5.0)

    prev = seen['prev']
    assert prev is not None and len(prev) == 4 and prev[0] is None
    for t in range(1, 4):
        assert prev[t].shape == (len(res['positions'][t - 1]), cfg['embedding_dim'])
    # frame 1's previous-frame descriptors come from window [0, 1], slot 0
    stack = pred._run_window(normalize_intensity(frames)[0:2])[1]
    yx = np.rint(res['positions'][0]).astype(int)
    assert np.allclose(prev[1], stack[0][:, yx[:, 0], yx[:, 1]].T, atol=1e-6)
