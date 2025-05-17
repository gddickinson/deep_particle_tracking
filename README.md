# Deep Particle Tracker

A Python application for deep learning-based particle tracking in microscopy images.

## Overview

Deep Particle Tracker is a comprehensive tool for detecting and tracking fluorescent particles in microscopy image sequences using deep learning. The application integrates:

- Simulation of realistic microscopy data
- Neural network training for particle detection
- Tracking of particles across frames
- Visualization of results

The application leverages the power of convolutional neural networks, particularly U-Net architectures with ConvLSTM layers, to process multiple frames simultaneously for improved detection and tracking.

## Features

- **Data Simulation**: Generate realistic particle data with various motion models, PSF types, and noise characteristics
- **Model Training**: Train deep learning models on simulated or experimental data
- **Prediction**: Apply trained models to detect particles and track them across frames
- **Visualization**: Interactive visualization of results with tracks and probability maps
- **User-Friendly GUI**: Intuitive interface for all operations

## Installation

### Requirements

- Python 3.7 or newer
- PyTorch 1.7 or newer
- PyQt5 for the GUI
- Various scientific Python packages (numpy, scipy, matplotlib, etc.)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deep_particle_tracker.git
   cd deep_particle_tracker
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install the package:
   ```
   pip install -e .
   ```

## Usage

### Starting the Application

Run the following command to start the application:

```
python main.py
```

Or if installed via pip:

```
deep_particle_tracker
```

### Command Line Options

- `--debug`: Enable debug logging
- `--cpu`: Force CPU mode (disable GPU)

### Using the GUI

The application has three main tabs:

1. **Simulation**: Create simulated particle data
2. **Training**: Train models on simulated or real data
3. **Prediction**: Apply trained models to new data

## Architecture

The software is organized into several modules:

- **simulator**: Particle, PSF, and noise simulation
- **models**: Neural network architectures and loss functions
- **training**: Data loading and model training
- **prediction**: Inference and tracking
- **visualization**: Result visualization tools
- **gui**: User interface components
- **utils**: Device management and threading utilities

## Examples

### Example 1: Simulating Data

1. Open the Simulation tab
2. Set simulation parameters (particle count, motion model, etc.)
3. Click "Simulate" to generate data
4. Review the results in the viewer
5. Save the simulation for later use

### Example 2: Training a Model

1. Open the Training tab
2. Select data source (simulated or from file)
3. Configure model and training parameters
4. Click "Start Training" to begin
5. Monitor training progress in real-time

### Example 3: Tracking Particles

1. Open the Prediction tab
2. Load your microscopy data
3. Select a trained model
4. Adjust detection and tracking parameters
5. Click "Run Prediction" to analyze
6. Explore the results with the interactive viewer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Deep-STORM paper by Nehme et al. for inspiration
- The PyTorch team for the deep learning framework
- The scientific Python community
