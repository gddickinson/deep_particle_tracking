"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests -------------------------------------------------------------------------------------

def test_info_counts():
    j = client.get("/api/info").json()
    assert (j["motions"], j["psfs"], j["noises"], j["networks"]) == (3, 2, 3, 3)
    assert j["weights_in_repo"] is False


def test_method_reads_source():
    j = client.get("/api/method").json()
    assert j["create_model"] == {"simple": "ParticleTrackerModel", "dual": "DualBranchParticleTracker",
                                 "attentive": "AttentiveParticleTracker"}
    names = {n["name"]: n for n in j["networks"]}
    assert names["ParticleTrackerModel"]["defaults"]["num_frames"] == 5
    assert "FocalLoss" in {x["name"] for x in j["losses"]}


def test_benchmark_is_labelled_baseline():
    j = client.get("/api/benchmark").json()
    assert "baseline" in j["method"] and "no trained weights" in j["deep_model"]
    assert len(j["snr_sweep"]["rows"]) == 21 and len(j["density_sweep"]["rows"]) == 6
    assert all(0 <= r["f1"] <= 1 for r in j["snr_sweep"]["rows"])


def test_score_matches_demo():
    d = client.get("/api/demo").json()
    r = client.post("/api/score", json={"seed": d["seed"], "sim": d["params"]}).json()
    for k in ("f1", "mota", "id_switches", "measured_snr"):
        assert abs(r[k] - (d["metrics"].get(k, d.get(k)))) < 1e-12


def test_score_rejects_unknown_parameter():
    assert client.post("/api/score", json={"seed": 1, "sim": {"colour": 1}}).status_code == 400


def test_models_local_only_and_predict_guarded():
    j = client.get("/api/models").json()
    assert "torch" in j and isinstance(j["checkpoints"], list)
    assert "/models" not in STATIC_ENDPOINTS
    r = client.post("/api/predict", json={"checkpoint": "../etc/passwd"})
    assert r.status_code in (404, 503)
