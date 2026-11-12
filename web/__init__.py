"""Web interface for Deep Particle Tracker (FastAPI + plain ES modules; see INTERFACE.md).

Contents:
- app.py           FastAPI app: /api routes, STATIC_ENDPOINTS exported for the GitHub Pages site
- benchmark.py     precomputed classical-baseline sweeps and the demo movie (Python results)
- method_info.py   network / loss facts read from the source with ast (no PyTorch needed)
- local_models.py  local-only: find checkpoints, run ParticlePredictor (needs PyTorch)
- static/          front end (index.html, pages, sim_engine.js / track_engine.js JS ports + node test)
"""
