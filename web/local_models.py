"""Local-only deep-model support for the web app: find checkpoints and run ParticlePredictor on a simulated movie.

Never used on the static site or in CI (no weights are committed; nothing is trained here). Needs PyTorch.
Checkpoints are looked for under checkpoints/ (where training/trainer.py saves them: <training_id>/best.pth,
latest.pth, ...) and models/ (PredictionManager's default models_dir).
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

from prediction.metrics import MetricsCalculator
from simulator.movie import gt_tracks, simulate_movie
from web.benchmark import MATCH_PX, score

ROOT = Path(__file__).resolve().parent.parent
SEARCH = ('checkpoints', 'models')


def torch_status() -> Dict:
    try:
        import torch  # noqa: F401
        return {'available': True, 'version': torch.__version__}
    except Exception as e:  # ImportError or a broken install
        return {'available': False, 'error': f'{type(e).__name__}: {e}'}


def find_checkpoints(root: Path = ROOT) -> List[Dict]:
    out = []
    for d in SEARCH:
        for p in sorted((root / d).rglob('*.pt*')) if (root / d).is_dir() else []:
            if p.suffix in ('.pth', '.pt'):
                out.append({'path': str(p.relative_to(root)), 'size_mb': round(p.stat().st_size / 1e6, 2)})
    return out


def predict(checkpoint: str, seed: int, sim: Dict, threshold: float = 0.5, nms_radius: int = 3,
            max_distance: float = 5.0, root: Path = ROOT) -> Dict:
    """Run the trained model and the classical baseline on the same seeded movie and score both."""
    path = (root / checkpoint).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(checkpoint)
    from prediction.inference import ParticlePredictor
    r = simulate_movie(seed=seed, **sim)
    pred = ParticlePredictor(str(path), threshold=threshold, nms_radius=nms_radius)
    out = pred.predict_sequence(r['frames'], link_particles=True, max_distance=max_distance)
    gt_pos = [g[:, 1:] for g in r['gt']]
    m = MetricsCalculator.calculate_all_metrics(out['positions'], gt_pos, out['tracks'] or [], gt_tracks(r), MATCH_PX)
    d, t = m['detection'], m.get('tracking', {})
    return {'checkpoint': checkpoint, 'seed': seed, 'params': r['params'],
            'model': {'precision': float(d['precision']), 'recall': float(d['recall']), 'f1': float(d['f1']),
                      'mota': float(t.get('mota', 0.0)), 'motp': float(t.get('motp', 0.0)),
                      'id_switches': int(t.get('id_switches', 0)), 'tracks': len(out['tracks'] or [])},
            'baseline': score(r),
            'detections': [np.asarray(p).astype(int).tolist() for p in out['positions']]}
