"""Seeded CPU comparison of tracking-loss settings (about 1-2 min): does localisation still learn, and do same-ID
embeddings still end up closer than different-ID ones?

Usage: python scripts/loss_balance_comparison.py [--steps 200]
Each setting trains the same seeded Simple model (base_filters 8, depth 2, 3-frame windows, embedding_dim 32) on the
same fresh simulated 32 x 32 px windows (4 per step, 6-12 particles, Adam lr 1e-3) and scores 16 held-out windows:
- loc: held-out localisation (focal) loss before -> after
- gap: (mean different-ID distance - mean same-ID distance) / different-ID distance, before -> after (> 0 = separated)
- init ratio: lambda_track * track_loss / localisation loss on the first batch (the full weight, ignoring warm-up)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models.losses import CombinedLoss, EmbeddingLoss  # noqa: E402
from models.network import create_model  # noqa: E402
from training.data_loader import SimulatedParticleDataset  # noqa: E402

T = 3
MODEL = dict(base_filters=8, depth=2, num_frames=T, embedding_dim=32)


def pair_gap(emb, track_ids):
    """(diff - same) / diff of mean embedding distances between particle pixels in different frames of each window."""
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
    s, d = float(torch.cat(same).mean()), float(torch.cat(diff).mean())
    return (d - s) / d


def run(label, steps, lambda_track, warmup, normalize_by_dim, seed=0):
    torch.manual_seed(seed)
    model = create_model('simple', **MODEL)
    ds = SimulatedParticleDataset(num_samples=10000, frame_size=(32, 32), frame_sequence_length=T,
                                  particles_per_frame=(6, 12))

    def batch(idx):
        items = [ds[i] for i in idx]
        return {k: torch.stack([it[k] for it in items]) for k in items[0]}

    held_out = batch(range(5000, 5016))
    loss_fn = CombinedLoss(track_loss=EmbeddingLoss(normalize_by_dim=normalize_by_dim),
                           lambda_track=lambda_track, track_warmup_steps=warmup)
    eval_fn = CombinedLoss(lambda_track=0.0)
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    def score():
        with torch.no_grad():   # BatchNorm on batch statistics (train mode), as in tests/test_embedding_training.py
            out = model(held_out['frames'])
            return float(eval_fn(out, held_out)['loc_loss']), pair_gap(out['embeddings'], held_out['track_ids'])

    loc0, gap0 = score()
    t0 = time.time()
    ratio = None
    for step in range(steps):
        b = batch(range(4 * step, 4 * step + 4))
        opt.zero_grad()
        losses = loss_fn(model(b['frames']), b)
        if ratio is None:
            ratio = lambda_track * float(losses['track_loss'].detach()) / float(losses['loc_loss'].detach())
        losses['total'].backward()
        opt.step()
        loss_fn.step()
    loc1, gap1 = score()
    print(f'{label:<38} init ratio {ratio:6.2f}  loc {loc0:.4f} -> {loc1:.4f}  gap {gap0:+.3f} -> {gap1:+.3f}'
          f'  ({time.time() - t0:.0f} s)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=50)
    a = ap.parse_args()
    run('localisation only (lambda 0)', a.steps, 0.0, 0, True)
    run('old: summed distance, lambda 0.5', a.steps, 0.5, 0, False)
    run('per-channel mean, lambda 0.5', a.steps, 0.5, 0, True)
    run(f'per-channel mean, lambda 0.5, warm-up {a.warmup}', a.steps, 0.5, a.warmup, True)
    run(f'per-channel mean, lambda 1.0, warm-up {a.warmup}', a.steps, 1.0, a.warmup, True)


if __name__ == '__main__':
    main()
