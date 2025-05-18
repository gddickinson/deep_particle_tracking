# Deep Particle Tracker User Manual

## Table of Contents

1. **Introduction**
   - Overview and Features
   - System Requirements
   - Installation

2. **Getting Started**
   - Interface Overview
   - Workflow Summary

3. **Particle Simulation**
   - Creating Simulated Datasets
   - Motion Models
   - PSF Models
   - Noise Models
   - Blinking Parameters
   - Saving Simulation Results

4. **Training Models**
   - Training Data Configuration
   - Model Architecture Selection
   - Training Parameters
   - Using Simulated Data
   - Incorporating Real Data
   - Monitoring Training Progress
   - Managing Multiple Training Sessions

5. **Testing and Prediction**
   - Loading Trained Models
   - Making Predictions on Test Data
   - Evaluating Model Performance
   - Fine-tuning for Better Results

6. **Working with Real Data**
   - Supported Data Formats
   - Data Preparation Guidelines
   - Importing Experimental Data
   - Transfer Learning with Real Data
   - Handling Different Imaging Conditions

7. **Analysis and Visualization**
   - Visualizing Tracking Results
   - Generating Animations
   - Analyzing Particle Trajectories
   - Exporting Results for Further Analysis

8. **Advanced Topics**
   - Custom Model Architectures
   - Hyperparameter Optimization
   - Performance Optimization
   - Large Dataset Handling

9. **Troubleshooting**
   - Common Issues and Solutions
   - Error Messages
   - Performance Issues

10. **References and Resources**
    - Algorithm References
    - Additional Tools
    - Citation Guidelines

---

## 1. Introduction

### Overview and Features

Deep Particle Tracker is a powerful application for simulating, detecting, and tracking fluorescent particles in microscopy images. It combines traditional computer vision techniques with state-of-the-art deep learning approaches to achieve robust tracking performance, especially in challenging conditions with low signal-to-noise ratios, high particle densities, and complex motion patterns.

**Key Features:**
- Simulation of particle dynamics with various motion models
- Support for different PSF (Point Spread Function) types
- Realistic noise simulation
- Multiple neural network architectures
- Training on both simulated and real data
- Comprehensive visualization tools
- Batch processing capabilities
- Performance metrics and analysis

### System Requirements

- **Operating System:** Windows 10/11, macOS 10.15+, or Linux
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** NVIDIA GPU with CUDA support or Apple Silicon with MPS (recommended)
- **Storage:** 2GB for installation, additional space for datasets
- **Python:** 3.8 or higher
- **Dependencies:** PyTorch, PyQt5, NumPy, SciPy, Matplotlib, tqdm

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deep-particle-tracker.git
   cd deep-particle-tracker
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python main.py
   ```

---

## 2. Getting Started

### Interface Overview

The Deep Particle Tracker application is organized into three main tabs:

- **Simulation Tab:** Generate synthetic particle data with configurable motion, PSF, and noise parameters.
- **Training Tab:** Train neural network models on simulated or real data.
- **Prediction Tab:** Use trained models to make predictions on new data and analyze results.

A console at the bottom of the window displays logging information and status messages.

### Workflow Summary

A typical workflow consists of the following steps:

1. **Simulation:** Generate synthetic particle data for training
2. **Training:** Train a model on the simulated data
3. **Validation:** Test the model on simulated data with different parameters
4. **Real-World Application:** Apply the model to real microscopy data
5. **Analysis:** Visualize and analyze tracking results

For users with existing data, steps 1-2 can be replaced with training directly on real data, optionally supplemented with simulated data for better generalization.

---

## 3. Particle Simulation

The Simulation tab allows you to generate synthetic particle datasets with realistic motion patterns, PSF models, and noise characteristics.

### Creating Simulated Datasets

To create a basic simulation:

1. Navigate to the **Simulation** tab
2. Set the **Image Size** (width and height in pixels)
3. Specify the **Number of Frames** to generate
4. Choose the **Number of Particles** to simulate
5. Click the **Simulate** button to run the simulation

The generated frames will appear in the right panel. Use the slider to browse through the frames.

### Motion Models

Deep Particle Tracker supports three motion models:

- **Brownian Motion:** Random walk model suitable for diffusing particles (e.g., molecules in solution)
  - Key parameter: **Diffusion Coefficient** (higher values lead to faster diffusion)

- **Directed Motion:** Motion with persistence in direction, suitable for active transport (e.g., motor proteins)
  - Key parameters: **Velocity** and **Direction Change Probability**

- **Confined Diffusion:** Brownian motion restricted to a confined region (e.g., membrane proteins)
  - Key parameters: **Diffusion Coefficient**, **Confinement Strength**, and **Confinement Radius**

To configure the motion model:

1. Select the motion model from the **Motion Model** dropdown
2. Set the parameters in the **Motion Parameters** group
3. The relevant parameters will appear based on your selection

### PSF Models

The Point Spread Function (PSF) determines how a point source appears in the image:

- **Gaussian:** Simple Gaussian approximation of the PSF
  - Key parameter: **Sigma** (standard deviation)

- **Airy Disk:** More accurate physical model based on diffraction
  - Key parameter: **Airy Radius** (first zero of the Airy pattern)

To configure the PSF model:

1. Select the PSF model from the **PSF Model** dropdown
2. Adjust the parameters in the **PSF Parameters** group

### Noise Models

Various noise sources can be simulated:

- **Poisson:** Shot noise from photon counting (signal-dependent)
- **Gaussian:** Electronic readout noise (signal-independent)
- **Poisson + Gaussian:** Combined model for realistic camera simulation

To configure noise:

1. Select the noise model from the **Noise Model** dropdown
2. Set the **Signal-to-Noise Ratio** (SNR)
3. Adjust the **Background Level** and **Read Noise** parameters as needed

### Blinking Parameters

For fluorophores that exhibit blinking behavior (e.g., fluorescent proteins, quantum dots):

1. Check the **Enable Blinking** option
2. Set the **On Probability** (probability of a dark fluorophore becoming bright)
3. Set the **Off Probability** (probability of a bright fluorophore becoming dark)

### Saving Simulation Results

To save your simulation:

1. Check the **Save Output** option
2. Click **Browse...** to select an output directory
3. After simulation, the results will be saved as:
   - TIFF stack for frames
   - NumPy files for positions and track IDs
   - JSON file for simulation parameters

**Tip:** The saved simulation data can be directly loaded in the Training tab.

---

## 4. Training Models

The Training tab allows you to train neural networks on simulated or real data.

### Training Data Configuration

Configure your training data source:

1. Select **Data Source**: 
   - **Simulated:** Generate new simulation data on-the-fly
   - **Load from File:** Use pre-existing data files

2. For simulated data, configure:
   - **Number of Samples**: How many sequences to generate
   - **Frames per Sequence**: Number of consecutive frames in each sequence
   - **Particles per Frame**: Range of particle counts
   - **Motion Model**: Type of motion to simulate

3. For file data, specify:
   - **Training Data**: Path to training data directory
   - **Validation Data**: (Optional) Path to validation data, otherwise, a subset of training data is used

### Model Architecture Selection

Select an appropriate model architecture:

- **Simple**: Basic encoder-decoder network for single-frame processing
- **Dual Branch**: Advanced architecture with separate branches for localization and tracking
- **Attentive**: Attention-based model for handling complex scenes and dense particle fields

Configure model parameters:

1. **Model Depth**: Number of encoding/decoding layers (deeper models can learn more complex patterns but require more computation)
2. **Base Filters**: Number of convolutional filters in the first layer (more filters increase model capacity)

### Training Parameters

Configure the training process:

1. **Batch Size**: Number of samples processed in each iteration (higher values use more memory)
2. **Epochs**: Number of complete passes through the training data
3. **Learning Rate**: Step size for optimizer (lower values provide more stable but slower training)
4. **Optimizer**: Algorithm for updating weights (Adam is a good default choice)
5. **Scheduler**: Learning rate adjustment strategy:
   - **None**: Constant learning rate
   - **Plateau**: Decrease when validation loss plateaus
   - **Cosine**: Smooth cyclic decrease
   - **Step**: Decrease by a factor at fixed intervals

6. **Training ID**: Unique identifier for this training session

### Using Simulated Data

To train on simulated data:

1. Select **Simulated** as the Data Source
2. Configure simulation parameters:
   - Increase **Number of Samples** for better generalization (1000+ recommended)
   - Set **Motion Model** to match your application
3. Configure model and training parameters
4. Click **Start Training**

**Best Practices:**
- Include a range of SNR values to improve robustness
- Vary particle counts to handle different densities
- Match simulation parameters to your experimental conditions when possible

### Incorporating Real Data

To train with real data:

1. Prepare your data in one of the following formats:
   - HDF5 files with 'frames', 'positions', and 'track_ids' datasets
   - Directory structure with:
     - TIFF stacks or sequential image files
     - positions.npy: Binary masks or coordinates
     - track_ids.npy: Track ID assignments (optional)

2. Select **Load from File** as the Data Source
3. Browse to select your prepared data directory
4. Configure model and training parameters
5. Click **Start Training**

**Best Practices for Real Data:**
- Split your data into training and validation sets
- Include examples with different imaging conditions
- Supplement with simulated data if real data is limited
- Annotate a subset of your data for ground truth positions and tracks

### Monitoring Training Progress

During training, you can monitor:

1. **Progress Bar**: Shows completion percentage
2. **Loss Plot**: Training and validation loss over epochs
3. **Active Trainings List**: All ongoing training sessions

To interact with training sessions:

- **Refresh**: Update the list and plots
- **Stop Training**: Terminate the current training session
- **Load Model**: Load a trained model for prediction

### Managing Multiple Training Sessions

The application supports multiple simultaneous training sessions:

1. Set a unique **Training ID** for each session
2. Start training
3. The session appears in the **Active Trainings** list
4. Select different sessions to view their progress
5. Models are saved in the `checkpoints/{training_id}` directory

**Tip:** Compare different architectures by training them simultaneously and comparing validation losses.

---

## 5. Testing and Prediction

The Prediction tab allows you to apply trained models to new data.

### Loading Trained Models

To load a trained model:

1. Navigate to the **Prediction** tab
2. Under **Model**, click **Browse...** to select a model file (*.pth)
3. Select the **Model Type** matching the architecture used for training
4. Set the **Detection Threshold** (typically 0.5, lower for recall, higher for precision)
5. Set the **NMS Radius** for non-maximum suppression (3-5 pixels is typical)

### Making Predictions on Test Data

To run predictions:

1. Under **Input Data**, select the data source:
   - **Load from File**: Use external data files
   - **Use Simulation**: Use data from the Simulation tab

2. If using a file, click **Browse...** to select the data file

3. Configure tracking parameters:
   - Check **Link Particles** to connect detections across frames
   - Set **Max Link Distance** (maximum distance particles can move between frames)

4. Click **Run Prediction**

5. View the results in the right panel:
   - Use the slider to browse through frames
   - Toggle visualization options:
     - **Show Positions**: Display detected particles
     - **Show Tracks**: Display particle trajectories
     - **Show Probability Map**: Display detection confidence

### Evaluating Model Performance

To evaluate model performance:

1. Run predictions on data with known ground truth
2. The console will display metrics including:
   - **Precision**: Fraction of detections that are correct
   - **Recall**: Fraction of true particles that are detected
   - **F1 Score**: Harmonic mean of precision and recall
   - For tracking: **MOTA** (Multiple Object Tracking Accuracy) and **ID Switches**

2. Compare results visually:
   - Green markers: Ground truth
   - Red markers: Predictions
   - Blue markers: Matched predictions

3. Save results by clicking **Save Results**

### Fine-tuning for Better Results

If initial results aren't satisfactory:

1. **Adjust the Detection Threshold**:
   - Lower for higher recall (fewer missed particles)
   - Higher for higher precision (fewer false positives)

2. **Modify the NMS Radius**:
   - Smaller for dense scenes
   - Larger for sparse scenes with larger particles

3. **Adjust Tracking Parameters**:
   - Increase **Max Link Distance** if tracks are fragmenting
   - Decrease if tracks are incorrectly connecting different particles

4. **Retrain with Adjusted Data**:
   - Add more examples similar to failure cases
   - Adjust simulation parameters to better match the test data
   - Include a wider range of conditions

5. **Try Different Model Architectures**:
   - Simple models for basic scenarios
   - Dual Branch for better tracking in challenging scenes
   - Attentive models for dense, complex environments

---

## 6. Working with Real Data

### Supported Data Formats

Deep Particle Tracker supports several input formats:

- **TIFF Stacks**: Multi-frame TIFF files
- **Image Sequences**: Numbered image files (PNG, JPEG, TIFF)
- **HDF5 Files**: With 'frames' dataset (and optionally 'positions' and 'track_ids')

For training data, additional information is required:

- **Position Ground Truth**: Binary masks or coordinate lists
- **Track ID Ground Truth**: (Optional) Track assignments for training tracking capability

### Data Preparation Guidelines

To prepare your data for optimal performance:

1. **Preprocessing**:
   - Normalize intensity to [0, 1] range
   - Remove background if possible
   - Correct for illumination non-uniformity
   - Consider denoising for very noisy images

2. **Ground Truth Annotation**:
   - For positions: Create binary masks or coordinate lists
   - For tracks: Assign unique IDs to particles across frames
   - Tools like ImageJ/Fiji with TrackMate can help with annotation

3. **Data Organization**:
   - Training set: Diverse examples covering expected conditions
   - Validation set: Examples similar but not identical to training
   - Test set: Examples representative of real application scenarios

4. **Recommended Structure**:
   ```
   data/
   ├── train/
   │   ├── frames.tif
   │   ├── positions.npy
   │   └── track_ids.npy
   ├── validation/
   │   ├── frames.tif
   │   ├── positions.npy
   │   └── track_ids.npy
   └── test/
       └── frames.tif
   ```

### Importing Experimental Data

To import your experimental data:

1. In the **Prediction** tab, select **Load from File**
2. Click **Browse...** and select your data file
3. The application will attempt to load and display the data
4. For prediction only, position and track ground truth are optional

### Transfer Learning with Real Data

Transfer learning allows you to leverage simulated data while adapting to real conditions:

1. **Pre-training**:
   - Train a model on a large simulated dataset
   - Use diverse simulation parameters to ensure robustness

2. **Fine-tuning**:
   - Navigate to the **Training** tab
   - Select **Load from File** and choose your real data
   - Browse to select your pre-trained model in **Training ID**
   - Reduce the **Learning Rate** (typically by factor of 10)
   - Set a smaller number of **Epochs** (10-20)
   - Click **Start Training**

3. **Domain Adaptation**:
   - If possible, create simulations that closely match your imaging conditions
   - Gradually mix in more real data as it becomes available
   - Monitor validation performance on real data

### Handling Different Imaging Conditions

To ensure robust performance across different imaging conditions:

1. **Match Simulation Parameters**:
   - Adjust PSF model and parameters to match your microscope
   - Set noise parameters based on camera specifications
   - Choose motion models appropriate for your particles

2. **Data Augmentation**:
   - In the **Training** tab, enable data augmentation
   - This applies random transformations to improve generalization
   - Particularly helpful when real data is limited

3. **Multiple Models Approach**:
   - Train separate models for significantly different conditions
   - Create a classifier to automatically select the appropriate model

---

## 7. Analysis and Visualization

### Visualizing Tracking Results

In the Prediction tab, you can visualize results in several ways:

1. **Frame View**:
   - Use the slider to browse through frames
   - Toggle **Show Positions** to display detected particles
   - Toggle **Show Tracks** to display particle trajectories
   - Toggle **Show Probability Map** to see detection confidence

2. **Probability Maps**:
   - Shows the confidence of detections
   - Brighter regions indicate higher confidence
   - Useful for identifying ambiguous regions

3. **Track Visualization**:
   - Colored lines show particle trajectories
   - Each track has a unique color
   - History length can be adjusted in the code

### Generating Animations

To create animations of tracking results:

1. Run prediction on your data
2. Click the **Animate** button
3. A new window will open with the animation
4. Controls for playback speed are available
5. Save the animation using the matplotlib controls

You can also save animation frames:

1. Click **Save Results** after running a prediction
2. Select a directory for output
3. Animation frames will be saved as image files in a subdirectory
4. These can be compiled into videos using external tools

### Analyzing Particle Trajectories

Advanced trajectory analysis can be performed by:

1. Exporting track data using **Save Results**
2. Using the exported NumPy files for custom analysis
3. Calculating metrics such as:
   - Mean Square Displacement (MSD)
   - Diffusion coefficients
   - Velocity profiles
   - Confinement indices

### Exporting Results for Further Analysis

To export results for external analysis:

1. After running a prediction, click **Save Results**
2. Select an output directory
3. The following files will be saved:
   - frames_{timestamp}.tif: Input frames
   - positions_{timestamp}.npy: Detected positions
   - tracks_{timestamp}.npy: Linked trajectories
   - probability_maps_{timestamp}.npy: Detection confidence maps

These files can be loaded in Python or MATLAB for custom analysis:

```python
# Python example
import numpy as np
import matplotlib.pyplot as plt

# Load tracks
tracks = np.load('tracks_20250517_141238.npy')

# Analyze track lengths
track_lengths = [np.sum(~np.isnan(track[:, 0])) for track in tracks]
plt.hist(track_lengths)
plt.xlabel('Track Length (frames)')
plt.ylabel('Count')
plt.show()
```

---

## 8. Advanced Topics

### Custom Model Architectures

Advanced users can extend the application with custom model architectures:

1. Create a new model class in `models/network.py`
2. Implement the required interfaces:
   - `__init__` method with appropriate parameters
   - `forward` method taking input tensors
   - Output format compatible with loss functions

3. Register the model in the `create_model` factory function
4. The model will become available in the UI

Example:

```python
class MyCustomModel(nn.Module):
    def __init__(self, input_channels=1, num_frames=5, base_filters=64):
        super().__init__()
        # Define your model architecture
        
    def forward(self, x):
        # Implement forward pass
        return output
        
# Add to create_model function
def create_model(model_type, **kwargs):
    if model_type == 'custom':
        return MyCustomModel(**kwargs)
    # Other model types...
```

### Hyperparameter Optimization

To systematically optimize model performance:

1. **Grid Search**:
   - Create a script to train models with different hyperparameter combinations
   - Use the trainer API programmatically:
   ```python
   from training.trainer import Trainer
   
   # Create trainer with different parameters
   for lr in [0.001, 0.0005, 0.0001]:
       for base_filters in [32, 64, 128]:
           trainer = Trainer(
               model_type='attentive',
               model_config={'base_filters': base_filters},
               lr=lr
           )
           trainer.train(train_loader, val_loader, epochs=30)
   ```

2. **Early Stopping**:
   - Monitor validation loss during training
   - Stop when loss stops improving
   - Use the `callbacks` parameter of the Trainer class:
   ```python
   def early_stopping(trainer, epoch, train_loss, val_loss):
       if epoch > 10 and val_loss > trainer.best_val_loss:
           return True  # Stop training
       return False
       
   trainer.callbacks = [early_stopping]
   ```

### Performance Optimization

To improve training and prediction speed:

1. **GPU Acceleration**:
   - Ensure your GPU is properly configured
   - Use the largest batch size that fits in memory
   - Enable GPU memory optimizations in PyTorch

2. **Data Loading**:
   - Increase `num_workers` in data loaders
   - Use memory-mapped files for large datasets
   - Pre-process and cache datasets

3. **Model Optimization**:
   - Reduce model depth or width for faster inference
   - Use half-precision (float16) for supported GPUs
   - Consider quantized models for deployment

### Large Dataset Handling

For very large datasets:

1. **Batch Processing**:
   - Split data into manageable chunks
   - Process each chunk separately
   - Combine results afterward

2. **Memory-Efficient Loading**:
   - Use the `chunks` parameter when loading HDF5 files
   - Process images sequentially rather than loading all at once
   - Implement a custom data loader with `__getitem__` that loads on-demand

3. **Distributed Processing**:
   - Implement multi-process data loading
   - Consider distributed training across multiple GPUs
   - Use parallel processing for CPU-bound operations

---

## 9. Troubleshooting

### Common Issues and Solutions

**Issue**: Simulation runs slowly
- **Solution**: Reduce image size or number of particles
- **Solution**: Disable complex PSF or noise models for preview purposes

**Issue**: Training loss doesn't decrease
- **Solution**: Decrease learning rate
- **Solution**: Check for data inconsistencies
- **Solution**: Try a different optimizer (Adam usually works well)

**Issue**: Model detects too many false positives
- **Solution**: Increase detection threshold
- **Solution**: Train with more diverse negative examples
- **Solution**: Use a more complex model architecture

**Issue**: Model misses particles
- **Solution**: Decrease detection threshold
- **Solution**: Train with more examples of difficult cases
- **Solution**: Ensure simulation parameters match real data

**Issue**: Tracks are fragmented
- **Solution**: Increase max link distance
- **Solution**: Improve detection performance
- **Solution**: Use a model with explicit tracking capabilities (Dual Branch or Attentive)

**Issue**: Predicting on larger images than training
- **Solution**: Use a fully convolutional architecture
- **Solution**: Implement sliding window inference

### Error Messages

**Error**: CUDA out of memory
- **Solution**: Reduce batch size
- **Solution**: Reduce model size (fewer base_filters)
- **Solution**: Process smaller image crops

**Error**: Cannot convert MPS Tensor to float64 dtype
- **Solution**: Ensure all tensors use float32 dtype
- **Solution**: Add explicit type conversions with `.float()`

**Error**: Tensor size mismatch in loss calculation
- **Solution**: Ensure model output and target have compatible shapes
- **Solution**: Check for dimension mismatches in data loading

**Error**: Value error with shape mismatches
- **Solution**: Ensure consistent tensor dimensions
- **Solution**: Check for correct channel dimension handling

### Performance Issues

**Issue**: GUI becomes unresponsive
- **Solution**: Reduce logging verbosity
- **Solution**: Use background threads for computation
- **Solution**: Update progress less frequently

**Issue**: High memory usage
- **Solution**: Clear variables explicitly when no longer needed
- **Solution**: Use smaller batches and process sequentially
- **Solution**: Release GPU memory with `torch.cuda.empty_cache()`

**Issue**: Slow training
- **Solution**: Use a GPU if available
- **Solution**: Optimize data loading pipeline
- **Solution**: Reduce data augmentation complexity

---

## 10. References and Resources

### Algorithm References

- **Particle Detection**:
  - U-Net architecture for segmentation
  - Focal Loss for imbalanced detection problems
  - Non-maximum suppression for peak finding

- **Particle Tracking**:
  - Linear assignment for frame-to-frame linking
  - Multiple hypothesis tracking for complex scenarios
  - Deep association metrics for appearance modeling

- **PSF Models**:
  - Gaussian approximation for computational efficiency
  - Airy disk model for physical accuracy
  - Gibson & Lanni model for 3D PSF

### Additional Tools

- **ImageJ/Fiji**: For manual annotation and ground truth creation
- **TrackMate**: ImageJ plugin for reference tracking
- **PySMLM**: Python library for Single Molecule Localization Microscopy
- **DeepTrack**: Alternative deep learning framework for comparison

### Citation Guidelines

If you use Deep Particle Tracker in your research, please cite:

```
@software{deep_particle_tracker,
  author = {Your Name},
  title = {Deep Particle Tracker: Neural Network Based Particle Tracking for Microscopy},
  year = {2025},
  url = {https://github.com/yourusername/deep-particle-tracker}
}
```

Additionally, please cite the specific algorithms and methods implemented in the software:

- U-Net architecture: Ronneberger et al., 2015
- Focal Loss: Lin et al., 2017
- ConvLSTM: Shi et al., 2015
