# Deep Particle Tracker: navigation map

Top-level packages (run from the repo root; imports are `from simulator.x import ...`).

| Path | Contents |
|---|---|
| `main.py` | Desktop entry point (PyQt5): `--debug`, `--cpu`, `--no_console`; opens `gui.main_window.MainWindow` |
| `gui/main_window.py` | `SimulationTab`, `TrainingTab`, `PredictionTab`, `MainWindow` (2.5k lines, over budget; ROADMAP split) |
| `gui/log_handler.py` | Qt log handler |
| `simulator/particle_generator.py` | `Particle`, `ParticleGenerator` (random particles, Brownian / directed / confined motion, blinking); 0.1 µm pixels, 0.1 s frames |
| `simulator/psf_models.py` | `GaussianPSF`, `AiryDiskPSF` (first zero at `airy_radius`), `AstigmaticPSF`, `EmpiricPSF` |
| `simulator/noise_models.py` | `PoissonNoise`, `GaussianNoise`, `MixedNoise`, `EMCCDNoise`, `sCMOSNoise`, `BackgroundNoise` |
| `simulator/motion_models.py` | `BrownianMotion`, `DirectedMotion`, `AnomalousDiffusion`, `ConfinedDiffusion`, `ActiveTransport` (class versions; the GUI uses ParticleGenerator) |
| `simulator/movie.py` | `simulate_movie(seed, **params)` = the Simulation tab pipeline (GUI and web call it); `gt_tracks`; `DEFAULTS` |
| `models/network.py` | `ParticleTrackerModel` (U-Net + ConvLSTM), `DualBranchParticleTracker`, `AttentiveParticleTracker`, `create_model` |
| `models/losses.py` | `FocalLoss`, `DistanceTransformLoss`, `EmbeddingLoss`, `HungarianMatchingLoss`, `CombinedLoss`, `TemporalEmbeddingLoss` |
| `training/data_loader.py` | `ParticleDataset`, `SimulatedParticleDataset` (its own copy of the simulation with random SNR per frame), transforms, `create_dataloaders` |
| `training/trainer.py` | `Trainer`, `TrainingManager` (checkpoints to `checkpoints/<id>/best.pth`) |
| `prediction/inference.py` | `ParticlePredictor`, `PredictionManager` (PyTorch); delegates to postprocess.py, re-exports `MetricsCalculator` |
| `prediction/postprocess.py` | `detect_peaks` (threshold + local max within `nms_radius`), `link_particles` (Hungarian, no gap closing) — torch-free |
| `prediction/metrics.py` | `MetricsCalculator` (precision/recall/F1, MOTA/MOTP, ID switches) — torch-free |
| `prediction/classical.py` | Classical baseline for the web app: `score_map` (matched filter in robust noise units), `run_baseline` |
| `visualization/visualization.py` | `ParticleVisualization`, `TrackingAnimator`, `TrackingVisualizer` (Qt), `TrainingProgressVisualization` |
| `utils/` | `DeviceManager` (`device_manager`), `ThreadManager` (`thread_manager`) |

## Web interface (`web/`, port 8201)
| Path | Contents |
|---|---|
| `run_web.py` | starts uvicorn on 127.0.0.1:8201 |
| `web/app.py` | FastAPI: `/api/info`, `/params`, `/method`, `/benchmark`, `/demo` (exported to the static site), POST `/score`; local-only `/models`, POST `/predict` |
| `web/benchmark.py` | precomputed baseline sweeps (SNR × noise model, density) and the seed-42 demo; `measured_snr` |
| `web/method_info.py` | network / loss names, docstrings and defaults read with `ast` (no PyTorch) |
| `web/local_models.py` | checkpoint discovery and `predict` (ParticlePredictor vs baseline on one movie) |
| `web/static/sim_engine.js` | JS port of `simulate_movie` (NumPy RNG via `np_random.js`, float32 steps) |
| `web/static/track_engine.js` | JS ports of score map, `detect_peaks`, `link_particles`, `MetricsCalculator`, `linear_sum_assignment`, gaussian/maximum filters |
| `web/static/scoring.js` | baseline + metrics on a JS movie (answers POST /score on the static site) |
| `web/static/{home,simulate,benchmark,method,model,about}.js` | pages; `sim_controls.js`, `sim_view.js`, `chart.js` helpers |
| `web/static/engine.test.mjs` | node test against `tests/fixtures/engine_fixture.json` (`scripts/make_engine_fixture.py`) |
| `scripts/build_static.py`, `scripts/screenshots.py` | static export for Pages; Playwright smoke test / README shots |
| `.github/workflows/pages.yml` | builds and deploys the static site on push to main |

## Tests (`tests/`)
`test_movie.py`, `test_airy_psf.py`, `test_detect_peaks.py`, `test_metrics.py`, `test_background_level.py`,
`test_classical.py`, `test_engine_fixture.py`, `test_web.py`, `test_local_models.py` (needs PyTorch; skipped otherwise).
