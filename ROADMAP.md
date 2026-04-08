# Deep Particle Tracker — Roadmap

## Current State
A well-structured deep learning application for particle tracking in microscopy images. The project has a modular architecture with separate packages for simulation, models, training, prediction, visualization, and GUI. Total ~10,500 lines of Python across ~20 files. Several files exceed 500 lines (`gui/main_window.py` at 2,644 lines is the largest offender). No test suite exists. The codebase uses PyTorch with U-Net/ConvLSTM architectures and has a PyQt5 GUI.

## Short-term Improvements
- [ ] Split `gui/main_window.py` (2,644 lines) into separate widget files (simulation_tab.py, training_tab.py, prediction_tab.py, toolbar.py)
- [ ] Split `visualization/visualization.py` (1,074 lines) into focused modules (track_viz.py, probability_map_viz.py, overlay_viz.py)
- [ ] Split `prediction/inference.py` (1,000 lines) into detector.py, tracker.py, and postprocessing.py
- [ ] Add type hints throughout — currently missing from most functions
- [ ] Add input validation for simulation parameters (particle count, SNR, frame count)
- [ ] Add proper error handling in `training/trainer.py` for GPU out-of-memory scenarios
- [ ] Add a `requirements.txt` version pinning (currently unpinned)
- [ ] Create unit tests for `simulator/` modules (motion_models, psf_models, noise_models) — these are pure-math and easy to test

## Feature Enhancements
- [ ] Add ONNX export support in `models/network.py` for deployment without PyTorch
- [ ] Add batch prediction mode in `prediction/inference.py` for processing multiple image stacks
- [ ] Add data augmentation options in `training/data_loader.py` (rotation, flipping, intensity jitter)
- [ ] Add a CLI interface alongside the GUI for headless/cluster usage
- [ ] Add model performance benchmarking (detection rate, tracking accuracy, RMSE) as a dedicated module
- [ ] Add support for importing real microscopy formats (TIFF stacks, ND2, CZI) beyond simulated data
- [ ] Implement track linking metrics (MOTA, MOTP) in prediction output

## Long-term Vision
- [ ] Add a REST API wrapper for integration with microscopy acquisition software (e.g., Micro-Manager)
- [ ] Support 3D particle tracking (z-stack volumes) — extend `models/network.py` to 3D convolutions
- [ ] Add multi-GPU training support with PyTorch DistributedDataParallel
- [ ] Package as a napari plugin for the scientific imaging community
- [ ] Add experiment tracking integration (MLflow or Weights & Biases) in `training/trainer.py`
- [ ] Create a pre-trained model zoo with checkpoints for common particle types

## Technical Debt
- [ ] `models/losses.py` (803 lines) contains too many loss functions — split into detection_losses.py and tracking_losses.py
- [ ] `training/data_loader.py` (917 lines) mixes data loading, augmentation, and dataset management — separate concerns
- [ ] `simulator/motion_models.py` (642 lines) could benefit from a base class pattern instead of if/else chains
- [ ] No logging configuration beyond basic debug flag — add structured logging with rotating file handlers
- [ ] Missing `__init__.py` docstrings in subpackages
- [ ] No CI/CD configuration — add GitHub Actions for linting and testing
