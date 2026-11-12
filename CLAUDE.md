# Deep Particle Tracker

@INTERFACE.md

- Packages are top-level (`simulator`, `models`, `training`, `prediction`, ...): run Python from the repo root.
- `simulator/movie.py` is the single simulation pipeline for the GUI and the web app; `web/static/sim_engine.js`
  must stay in step (regenerate `tests/fixtures/engine_fixture.json` with `scripts/make_engine_fixture.py`).
- No trained weights are committed (`*.pth`, `*.pt` are ignored); never train in CI.
- Tests: `python -m pytest -q tests` and `node web/static/engine.test.mjs`. Session notes: SESSION_LOG.md.
