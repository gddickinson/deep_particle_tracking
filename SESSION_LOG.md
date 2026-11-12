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

## 2026-11-12 (follow-up): SNR, sequences, embeddings, 16-bit input
- One SNR definition (`simulator/movie.py` noise_levels): peak signal (intensity 1) above background / noise SD at
  the background; Poisson: sqrt(background + read noise^2) photons. Simulator, GUI and SimulatedParticleDataset use
  it; measured SNR matches the setting at 1, 3, 10 (Poisson + Gaussian capped at 100/3 by 3 counts of read noise).
  Background Level widget removed. JS simulator and fixtures regenerated; benchmark recomputed.
- Predictor feeds sliding windows of the trained length (ConvLSTM sees T > 1); embeddings returned by the simple
  model and used for linking (embedding distance + position gate); 16-bit input normalised by percentiles.
- Found while testing: the simple model output logits into BCE (training crashed on step 1) -> sigmoid; the loss
  compared predictions (last frame) with the first target frame -> last frame.
- Still open: DualBranch / Attentive models crash on any forward pass (DecoderBlock called without skip_channels)
  and their 8x super-resolution output is not mapped back to frame pixels; EmbeddingLoss needs >= 2 frames of
  embeddings, so with one output frame the tracking loss is 0 and embeddings are not trained.

## 2026-11-12 (follow-up 2): Dual Branch and Attentive models
- Fixed both crashing on every forward pass (DecoderBlock skip_channels), their super-resolution output
  (upsample_factor ignored; loss and predictor now handle k x maps: max-pool for the loss, sub-pixel peaks mapped
  to frame pixels), and Attentive decoding the central instead of the last frame. tests/test_models_dual_attentive.py.
- Still open for George: EmbeddingLoss compares embeddings across frames but every model outputs one frame of
  embeddings, so the tracking loss is 0 and embeddings are untrained (model output shape left unchanged).

## 2026-11-12 (follow-up 3): embeddings trained; over-budget files split
- Refactor (no behaviour change): models/network.py (779 lines) -> blocks.py + tracker_simple/dual/attentive.py;
  models/losses.py (810) -> loss_localization/embedding/matching.py + CombinedLoss; prediction/inference.py (510) ->
  predictor.py + manager.py. Old modules re-export everything; seeded outputs, losses and state_dict keys identical.
  web/method_info.py follows the re-exports (same /api/method content).
- Fixed: embeddings were never trained (one embedding frame -> EmbeddingLoss had no frame pairs -> 0). All three
  models decode every input frame (decode_frames, frames folded into the batch) and return embeddings
  (B, T, E, H, W); positions unchanged (last frame; identical in eval mode). CombinedLoss pairs them by the
  simulator's track_ids. The predictor links t-1 -> t with both frames' embeddings from the window ending at t.
  tests/test_embedding_training.py: loss > 0 with gradients at the embedding head (all models); seeded 200-step CPU
  run: held-out same-ID pairs become clearly closer than different-ID pairs; Trainer epoch runs for all models.
- For George: the tracking term (lambda_track 0.5) was 0 until now and starts ~10-60x the localisation loss on an
  untrained model; its weight may want tuning once real training runs are made.


## 2026-11-12 (follow-up 4): tracking-loss balance
- EmbeddingLoss now averages squared embedding distances over channels (`normalize_by_dim=True`), so its scale no
  longer depends on embedding_dim or embedding magnitude: at initialisation 0.5 x track / loc is 0.9-4.6x across
  the three models and two particle densities (was 2-44x with summed distances).
- CombinedLoss ramps the tracking weight linearly from 0 to lambda_track (default 0.5) over track_warmup_steps
  (default 200, ~1.6 epochs of the GUI's default 1000 samples / batch 8); validation (eval mode) always uses the
  full weight so best-checkpoint selection is not biased by the ramp. Trainer takes track_warmup_steps, steps the
  schedule after each optimiser step and saves/restores its position in checkpoints. Defaults live in
  models/losses.py (DEFAULT_LAMBDA_TRACK, DEFAULT_TRACK_WARMUP_STEPS); /api/method reads them.
- scripts/loss_balance_comparison.py (seeded CPU, 200 steps, ~80 s): localisation still learns and same/different-ID
  separation still appears with the new defaults (numbers in the web-interface report). tests/test_loss_balance.py.
