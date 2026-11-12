# Session log

## 2026-11-12: web interface + static site (branch web-interface)
- Factored the Simulation tab pipeline into `simulator/movie.py` (GUI calls it; identical frames vs the old inline
  code for all motion/PSF/noise models), detection/linking into `prediction/postprocess.py` and `MetricsCalculator`
  into `prediction/metrics.py` (torch-free; `prediction/__init__` imports PyTorch classes lazily).
- Fixed (each with a regression test, separate commits): Airy PSF first zero at 0.61 x airy_radius; NMS radius used as
  window width; detection metrics matching on unthresholded distances (lost TPs); ID switches missed after a frame
  without a match; Background Level never used; README clone URL.
- Checked for the sibling-repo bugs (MSD lags counted in rows, length-dependent confinement ratio): not present; this
  repo has no MSD or confinement analysis.
- Added a classical baseline (`prediction/classical.py`) because no trained weights are in the repo.
- Web app (port 8201): Home, Simulator (JS port, NumPy RNG reproduced: same movie per seed), Benchmark (Python
  sweeps), Method (read from source with ast), Trained model (local only, PyTorch + your checkpoint), About.
- Open questions for George: see reports (SNR setting semantics, predictor single-frame inference, embeddings unused,
  16-bit input normalisation).
