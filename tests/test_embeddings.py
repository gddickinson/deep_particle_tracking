"""The network returns its embeddings and 'embedding' linking uses them (mechanics, not accuracy)."""
import numpy as np
import pytest

from prediction.postprocess import link_particles, link_particles_with_embeddings


def test_embedding_cost_decides_within_the_gate():
    # Track A at x=10 continues; two candidates both within the 5 px gate: the nearer one (x=11) has a very
    # different embedding, the farther one (x=13) the same embedding as A. Position linking takes x=11,
    # embedding linking x=13. A candidate outside the gate is never linked.
    pos = [np.array([[10.0, 10.0]]), np.array([[10.0, 11.0], [10.0, 13.0], [10.0, 30.0]]), np.array([[10.0, 13.5]])]
    desc = [np.array([[1.0, 0.0]]), np.array([[0.0, 5.0], [1.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0]])]
    by_pos = link_particles(pos, 5.0, min_length=2)
    by_emb = link_particles_with_embeddings(pos, desc, 5.0, min_length=2)
    assert by_pos[0][1].tolist() == [10.0, 11.0]
    assert by_emb[0][1].tolist() == [10.0, 13.0] and by_emb[0][2].tolist() == [10.0, 13.5]
    assert all(not (tr[1] == [10.0, 30.0]).all() or np.isnan(tr[0, 0]) for tr in by_emb)


def test_tiny_model_returns_and_uses_embeddings(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from models.network import create_model
    from prediction import predictor as inference  # ParticlePredictor lives in prediction/predictor.py
    from prediction.inference import ParticlePredictor
    cfg = {'base_filters': 4, 'depth': 2, 'num_frames': 3, 'embedding_dim': 6}
    model = create_model('simple', **cfg)
    out = model(torch.rand(2, 3, 1, 32, 32))
    assert set(out) == {'positions', 'embeddings'}
    assert tuple(out['positions'].shape) == (2, 1, 1, 32, 32) and tuple(out['embeddings'].shape) == (2, 1, 6, 32, 32)

    path = tmp_path / 'best.pth'
    torch.save({'model_state_dict': model.state_dict(), 'model_type': 'simple', 'model_config': cfg}, path)
    pred = ParticlePredictor(str(path), threshold=-1e9, nms_radius=3)   # untrained: accept every local maximum
    calls = []
    real = inference.link_particles_with_embeddings
    monkeypatch.setattr(inference, 'link_particles_with_embeddings',
                        lambda p, d, m, *a: calls.append([x.shape for x in d]) or real(p, d, m, *a))
    frames = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
    res = pred.predict_sequence(frames, max_distance=5.0)
    assert len(res['embeddings']) == 4 and res['embeddings'][0].shape == (6, 32, 32)
    assert calls and all(s[1] == 6 for s in calls[0])            # one 6-vector per detection
    assert [s[0] for s in calls[0]] == [len(p) for p in res['positions']]
    assert isinstance(res['tracks'], list)
