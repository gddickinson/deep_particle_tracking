"""Write tests/fixtures/engine_fixture.json: Python results that the JS engines must reproduce.

Covers simulator/movie.py simulate_movie (seeded movies for every motion / PSF / noise model, with blinking),
prediction/classical.py (score map, baseline detections and tracks), prediction/metrics.py, the scipy
linear_sum_assignment used by linking and metrics, and scipy.special.j1 for the Airy PSF.

Run: python scripts/make_engine_fixture.py   (tests/test_engine_fixture.py checks the file is current)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy import special  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from prediction.classical import run_baseline, score_map  # noqa: E402
from prediction.metrics import MetricsCalculator  # noqa: E402
from simulator.movie import simulate_movie, gt_tracks  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_fixture.json"

MOVIES = [
    (1, dict(width=24, height=20, num_frames=5, num_particles=6)),
    (2, dict(width=24, height=20, num_frames=5, num_particles=6, motion='Directed', velocity=3.0, direction_change=0.5,
             noise='Gaussian', snr=40.0)),
    (3, dict(width=24, height=20, num_frames=5, num_particles=6, motion='Confined', diffusion=0.5,
             confinement_strength=5.0, confinement_radius=3.0, noise='Poisson + Gaussian', blinking=True)),
    (4, dict(width=24, height=20, num_frames=4, num_particles=4, psf='Airy Disk', airy_radius=2.0, psf_sigma=1.5,
             blinking=True, on_prob=0.5, off_prob=0.5, snr=3.0)),
    (5, dict(width=16, height=16, num_frames=3, num_particles=0, background=0.2)),
    (6, dict(width=30, height=30, num_frames=4, num_particles=8, motion='Directed', velocity=8.0, noise='Poisson + Gaussian',
             read_noise=0.0, psf_sigma=2.0)),
]
BASELINE = [
    (7, dict(width=48, height=40, num_frames=8, num_particles=10, snr=10.0), {}),
    (8, dict(width=48, height=40, num_frames=8, num_particles=12, blinking=True, noise='Poisson + Gaussian'),
     dict(threshold=4.0, nms_radius=2, max_distance=4.0)),
]


def nan_list(a):
    return [[None if np.isnan(v) else float(v) for v in row] for row in np.asarray(a, float)]


def metrics_json(m):
    return json.loads(json.dumps(m, default=float))


def build():
    out = {"movies": [], "baseline": [], "lsap": [], "j1": []}
    for seed, p in MOVIES:
        r = simulate_movie(seed=seed, **p)
        out["movies"].append({"seed": seed, "params": p, "frames": [f.astype(float).ravel().tolist() for f in r["frames"]],
                              "gt": [g.tolist() for g in r["gt"]]})
    for seed, p, bp in BASELINE:
        r = simulate_movie(seed=seed, **p)
        b = run_baseline(r["frames"], **bp)
        gt_pos = [g[:, 1:] for g in r["gt"]]
        m = MetricsCalculator.calculate_all_metrics(b["positions"], gt_pos, b["tracks"], gt_tracks(r), 5.0)
        out["baseline"].append({"seed": seed, "params": p, "baseline_params": bp,
                                "score0": score_map(r["frames"][0], b["params"]["sigma"]).ravel().tolist(),
                                "positions": [q.tolist() for q in b["positions"]],
                                "tracks": [nan_list(t) for t in b["tracks"]],
                                "gt_tracks": [nan_list(t) for t in gt_tracks(r)], "metrics": metrics_json(m)})
    rng = np.random.default_rng(0)
    for nr, nc in [(3, 3), (4, 6), (7, 5), (1, 4), (6, 1), (9, 9)]:
        c = np.round(rng.uniform(0, 10, (nr, nc)), 3)
        c[c > 7] = 1e6
        a, b = linear_sum_assignment(c)
        out["lsap"].append({"cost": c.tolist(), "rows": a.tolist(), "cols": b.tolist()})
    xs = np.concatenate([np.linspace(1e-6, 30, 301), [11.9, 12.0, 12.1, 50.0, 200.0, 517.3]])
    out["j1"] = [[float(x), float(special.j1(x))] for x in xs]
    return out


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
