"""FastAPI web app for Deep Particle Tracker: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8201, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.
/models and /predict (the trained deep model) are local-only: they need PyTorch and a checkpoint on this machine.
"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from prediction.classical import BASELINE_DEFAULTS
from simulator.movie import DEFAULTS, MOTIONS, NOISES, PSFS, simulate_movie
from web.benchmark import benchmark, demo, measured_snr, score
from web.local_models import find_checkpoints, predict, torch_status
from web.method_info import method_info

STATIC = Path(__file__).resolve().parent / "static"
DESCRIPTION = "A Python application for deep learning-based particle tracking in microscopy images."

app = FastAPI(title="Deep Particle Tracker")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Slider ranges for the web simulator: the Simulation tab's own ranges, except image size and frame/particle counts,
# which are kept small so the movie is generated quickly in the browser.
RANGES = {
    'width': [32, 256], 'height': [32, 256], 'num_frames': [2, 100], 'num_particles': [0, 300],
    'diffusion': [0.01, 10.0], 'velocity': [0.1, 10.0], 'direction_change': [0.0, 1.0],
    'confinement_strength': [0.1, 10.0], 'confinement_radius': [1.0, 100.0],
    'psf_sigma': [0.5, 5.0], 'airy_radius': [0.5, 5.0], 'snr': [1.0, 50.0],
    'read_noise': [0.0, 20.0], 'on_prob': [0.0, 1.0], 'off_prob': [0.0, 1.0],
}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary and counts for the home page."""
    m = method_info()
    return {"title": "Deep Particle Tracker", "description": DESCRIPTION,
            "motions": len(MOTIONS), "psfs": len(PSFS), "noises": len(NOISES),
            "networks": len(m["create_model"]), "losses": len(m["losses"]),
            "weights_in_repo": False}


@app.get("/api/params")
def params():
    """Simulator defaults (Simulation tab), choices, slider ranges and the baseline settings."""
    return {"defaults": DEFAULTS, "motions": MOTIONS, "psfs": PSFS, "noises": NOISES, "ranges": RANGES,
            "baseline": BASELINE_DEFAULTS, "pixel_um": 0.1, "frame_s": 0.1}


@app.get("/api/method")
def method():
    return method_info()


@app.get("/api/benchmark")
def get_benchmark():
    """Classical-baseline scores over SNR and density sweeps (precomputed; no deep-model weights in the repo)."""
    return benchmark()


@app.get("/api/demo")
def get_demo():
    """One seeded movie: ground truth, baseline detections/tracks and scores from the Python code."""
    return demo()


class SimRequest(BaseModel):
    seed: int = 1
    sim: dict = {}
    baseline: dict = {}


@app.post("/api/score")
def post_score(req: SimRequest):
    """Python scores for any seeded movie (the static site computes the same in the browser)."""
    try:
        r = simulate_movie(seed=req.seed, **req.sim)
        return {**score(r, **req.baseline), "measured_snr": measured_snr(r)}
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/models")
def models():
    """Local only: is PyTorch importable, and which checkpoints are on this machine."""
    return {"torch": torch_status(), "checkpoints": find_checkpoints()}


class PredictRequest(BaseModel):
    checkpoint: str
    seed: int = 1
    sim: dict = {}
    threshold: float = 0.5
    nms_radius: int = 3
    max_distance: float = 5.0


@app.post("/api/predict")
def post_predict(req: PredictRequest):
    """Local only: run a trained checkpoint and the classical baseline on the same simulated movie."""
    if not torch_status()["available"]:
        raise HTTPException(503, "PyTorch is not importable in this Python environment")
    try:
        return predict(req.checkpoint, req.seed, req.sim, req.threshold, req.nms_radius, req.max_distance)
    except FileNotFoundError:
        raise HTTPException(404, f"checkpoint not found: {req.checkpoint}")
    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


# GET paths (without the /api prefix) written to the static site as api/<path>.json. /models is local-only.
STATIC_ENDPOINTS = ["/info", "/params", "/method", "/benchmark", "/demo"]
