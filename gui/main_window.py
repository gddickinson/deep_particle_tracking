"""
Main GUI application for Deep Particle Tracker.
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import datetime
import json
import tifffile
from PIL import Image

import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QPushButton, QLabel, QLineEdit,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QFileDialog,
    QGroupBox, QScrollArea, QSplitter, QProgressBar, QSlider,
    QTextEdit, QToolButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QMessageBox, QListWidget, QListWidgetItem,
    QFormLayout, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMutex, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor

from utils.device_manager import device_manager
from utils.thread_manager import thread_manager
from simulator.particle_generator import ParticleGenerator
from simulator.psf_models import GaussianPSF, AiryDiskPSF
from simulator.noise_models import PoissonNoise, GaussianNoise, MixedNoise
from simulator.motion_models import BrownianMotion, DirectedMotion, ConfinedDiffusion
from training.data_loader import create_dataloaders, SimulatedParticleDataset
from training.trainer import TrainingManager
from prediction.inference import PredictionManager
from visualization.visualization import (
    ParticleVisualization, TrackingAnimator, TrackingVisualizer,
    TrainingProgressVisualization, MatplotlibCanvas
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsoleLogHandler(logging.Handler):
    """Logging handler to redirect logs to a QTextEdit widget with crash prevention."""

    def __init__(self, console_widget, max_lines=1000, rate_limit=100):
        """
        Initialize the console log handler with crash prevention.

        Args:
            console_widget: QTextEdit widget to display logs
            max_lines: Maximum number of lines to keep in console
            rate_limit: Maximum number of log messages per second
        """
        super().__init__()
        self.console = console_widget
        self.max_lines = max_lines
        self.rate_limit = rate_limit
        self.message_count = 0
        self.last_flush_time = time.time()
        self.message_queue = []
        self.mutex = QMutex()  # Thread safety for message queue

        # Set up a timer to periodically flush messages to console
        self.flush_timer = QTimer()
        self.flush_timer.timeout.connect(self.periodic_flush)
        self.flush_timer.start(100)  # Flush every 100ms

        # Keep track of repeated messages
        self.last_message = None
        self.repeat_count = 0

        # Warning types that should be limited
        self.limited_warnings = set()

        # Initialize console with clear message
        self.console.clear()
        self.console.append('<font color="blue">Logging initialized. System ready.</font>')

    def emit(self, record):
        """
        Queue a log record for display.

        Args:
            record: Log record
        """
        # Check if this is a shape warning we should limit
        if (record.levelno == logging.WARNING and
            ('shape' in record.getMessage() or 'dimension' in record.getMessage())):
            warning_key = f"{record.name}_{record.funcName}_{record.lineno}"

            # If we've seen this warning too many times, skip it
            if warning_key in self.limited_warnings and len(self.limited_warnings) > 10:
                return

            # Add to limited warnings set
            self.limited_warnings.add(warning_key)

        # Format the message
        msg = self.format(record)

        # Check for identical repeated messages
        if msg == self.last_message:
            self.repeat_count += 1
            if self.repeat_count > 3:  # Only show up to 3 identical messages in a row
                return
        else:
            self.last_message = msg
            self.repeat_count = 1

        # Add timestamp and level
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        level_name = record.levelname

        # Format with appropriate colors
        color = 'black'
        if record.levelno >= logging.ERROR:
            color = 'red'
        elif record.levelno >= logging.WARNING:
            color = 'orange'
        elif record.levelno >= logging.INFO:
            color = 'blue'

        formatted_msg = f'<font color="{color}">[{timestamp}] [{level_name}] {msg}</font>'

        # Queue the message
        self.mutex.lock()
        self.message_queue.append(formatted_msg)
        self.mutex.unlock()

        # If too many messages are queued, force a flush
        if len(self.message_queue) > 50:
            self.periodic_flush()

    def periodic_flush(self):
        """Periodically flush queued messages to the console."""
        # Skip if no messages or we've already flushed recently
        if not self.message_queue:
            return

        current_time = time.time()
        elapsed = current_time - self.last_flush_time

        # Rate limit - don't flush too often
        if elapsed < 0.1 and len(self.message_queue) < 10:  # 10 Hz max unless queue is big
            return

        self.mutex.lock()
        messages = self.message_queue.copy()
        self.message_queue.clear()
        self.mutex.unlock()

        # If too many messages, trim to just the first few and last few
        if len(messages) > self.rate_limit:
            # Keep first 10 and last 10 messages
            tmp = messages[:10]
            tmp.append(f'<font color="purple">... {len(messages) - 20} messages skipped ...</font>')
            tmp.extend(messages[-10:])
            messages = tmp

        # Batch append messages to console
        html = '<br>'.join(messages)
        self.console.append(html)

        # Trim console text if it gets too long
        self._trim_console()

        # Scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Update flush time
        self.last_flush_time = current_time

    def _trim_console(self):
        """Trim console text to prevent it from getting too large."""
        # Get current text
        text = self.console.toPlainText()
        lines = text.split('\n')

        # Check if we need to trim
        if len(lines) > self.max_lines:
            # Keep only the last max_lines
            lines_to_keep = lines[-self.max_lines:]

            # Set new text
            new_text = '\n'.join(lines_to_keep)

            # Temporarily disconnect scrollbar to prevent crash
            scrollbar = self.console.verticalScrollBar()
            scrollbar_pos = scrollbar.value()

            # Update text
            self.console.clear()
            self.console.append('<font color="purple">... earlier messages removed ...</font><br>')
            self.console.append('<br>'.join(lines_to_keep))


class SimulationTab(QWidget):
    """Tab for simulating particle data."""

    simulation_done = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize the simulation tab."""
        super().__init__(parent)

        # Set up UI
        self.setup_ui()

        # Initialize simulation parameters
        self.frames = None
        self.positions = None
        self.track_ids = None

        # Initialize particle generator
        self.particle_generator = ParticleGenerator()

        # Connect signals
        self.setup_connections()

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QHBoxLayout(self)

        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Simulation parameters group
        sim_group = QGroupBox("Simulation Parameters")
        sim_layout = QFormLayout(sim_group)

        # Image size
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 2048)
        self.width_spin.setValue(512)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 2048)
        self.height_spin.setValue(512)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Width:"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("Height:"))
        size_layout.addWidget(self.height_spin)

        sim_layout.addRow("Image Size:", size_layout)

        # Number of frames
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(1, 1000)
        self.num_frames_spin.setValue(10)
        sim_layout.addRow("Number of Frames:", self.num_frames_spin)

        # Number of particles
        self.num_particles_spin = QSpinBox()
        self.num_particles_spin.setRange(1, 1000)
        self.num_particles_spin.setValue(20)
        sim_layout.addRow("Number of Particles:", self.num_particles_spin)

        # Motion model
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(["Brownian", "Directed", "Confined"])
        sim_layout.addRow("Motion Model:", self.motion_combo)

        # Motion parameters
        self.motion_params_group = QGroupBox("Motion Parameters")
        self.motion_params_layout = QFormLayout(self.motion_params_group)

        # Brownian motion parameters
        self.diffusion_coef_spin = QDoubleSpinBox()
        self.diffusion_coef_spin.setRange(0.01, 10.0)
        self.diffusion_coef_spin.setValue(0.1)
        self.diffusion_coef_spin.setSingleStep(0.01)
        self.motion_params_layout.addRow("Diffusion Coefficient:", self.diffusion_coef_spin)

        # Directed motion parameters
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0.1, 10.0)
        self.velocity_spin.setValue(1.0)
        self.velocity_spin.setSingleStep(0.1)
        self.motion_params_layout.addRow("Velocity:", self.velocity_spin)

        self.direction_change_spin = QDoubleSpinBox()
        self.direction_change_spin.setRange(0.0, 1.0)
        self.direction_change_spin.setValue(0.1)
        self.direction_change_spin.setSingleStep(0.01)
        self.motion_params_layout.addRow("Direction Change Prob:", self.direction_change_spin)

        # Confined motion parameters
        self.confinement_strength_spin = QDoubleSpinBox()
        self.confinement_strength_spin.setRange(0.1, 10.0)
        self.confinement_strength_spin.setValue(1.0)
        self.confinement_strength_spin.setSingleStep(0.1)
        self.motion_params_layout.addRow("Confinement Strength:", self.confinement_strength_spin)

        self.confinement_radius_spin = QDoubleSpinBox()
        self.confinement_radius_spin.setRange(1.0, 100.0)
        self.confinement_radius_spin.setValue(10.0)
        self.confinement_radius_spin.setSingleStep(1.0)
        self.motion_params_layout.addRow("Confinement Radius:", self.confinement_radius_spin)

        sim_layout.addRow(self.motion_params_group)

        # PSF parameters
        self.psf_group = QGroupBox("PSF Parameters")
        psf_layout = QFormLayout(self.psf_group)

        self.psf_combo = QComboBox()
        self.psf_combo.addItems(["Gaussian", "Airy Disk"])
        psf_layout.addRow("PSF Model:", self.psf_combo)

        self.psf_sigma_spin = QDoubleSpinBox()
        self.psf_sigma_spin.setRange(0.5, 5.0)
        self.psf_sigma_spin.setValue(1.0)
        self.psf_sigma_spin.setSingleStep(0.1)
        psf_layout.addRow("Sigma:", self.psf_sigma_spin)

        self.airy_radius_spin = QDoubleSpinBox()
        self.airy_radius_spin.setRange(0.5, 5.0)
        self.airy_radius_spin.setValue(1.22)
        self.airy_radius_spin.setSingleStep(0.1)
        psf_layout.addRow("Airy Radius:", self.airy_radius_spin)

        sim_layout.addRow(self.psf_group)

        # Noise parameters
        self.noise_group = QGroupBox("Noise Parameters")
        noise_layout = QFormLayout(self.noise_group)

        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["Poisson", "Gaussian", "Poisson + Gaussian"])
        noise_layout.addRow("Noise Model:", self.noise_combo)

        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(1.0, 50.0)
        self.snr_spin.setValue(10.0)
        self.snr_spin.setSingleStep(0.5)
        noise_layout.addRow("Signal-to-Noise Ratio:", self.snr_spin)

        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0.0, 1.0)
        self.background_spin.setValue(0.1)
        self.background_spin.setSingleStep(0.01)
        noise_layout.addRow("Background Level:", self.background_spin)

        self.read_noise_spin = QDoubleSpinBox()
        self.read_noise_spin.setRange(0.0, 20.0)
        self.read_noise_spin.setValue(3.0)
        self.read_noise_spin.setSingleStep(0.1)
        noise_layout.addRow("Read Noise:", self.read_noise_spin)

        sim_layout.addRow(self.noise_group)

        # Blinking parameters
        self.blinking_group = QGroupBox("Blinking Parameters")
        blinking_layout = QFormLayout(self.blinking_group)

        self.blinking_check = QCheckBox("Enable Blinking")
        blinking_layout.addRow(self.blinking_check)

        self.on_prob_spin = QDoubleSpinBox()
        self.on_prob_spin.setRange(0.0, 1.0)
        self.on_prob_spin.setValue(0.7)
        self.on_prob_spin.setSingleStep(0.05)
        blinking_layout.addRow("On Probability:", self.on_prob_spin)

        self.off_prob_spin = QDoubleSpinBox()
        self.off_prob_spin.setRange(0.0, 1.0)
        self.off_prob_spin.setValue(0.2)
        self.off_prob_spin.setSingleStep(0.05)
        blinking_layout.addRow("Off Probability:", self.off_prob_spin)

        sim_layout.addRow(self.blinking_group)

        # Output parameters
        self.output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(self.output_group)

        self.save_check = QCheckBox("Save Output")
        output_layout.addRow(self.save_check)

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_path)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.output_path_edit)
        path_layout.addWidget(browse_button)
        output_layout.addRow("Output Path:", path_layout)

        sim_layout.addRow(self.output_group)

        # Simulation buttons
        sim_buttons_layout = QHBoxLayout()
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.run_simulation)
        sim_buttons_layout.addWidget(self.simulate_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_simulation)
        sim_buttons_layout.addWidget(self.clear_button)

        # Add to left panel
        left_layout.addWidget(sim_group)
        left_layout.addLayout(sim_buttons_layout)
        left_layout.addStretch()

        # Right panel (visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Frame viewer
        self.frame_viewer_group = QGroupBox("Frame Viewer")
        frame_viewer_layout = QVBoxLayout(self.frame_viewer_group)

        # Matplotlib canvas for frame display
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.canvas = MatplotlibCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Simulated Frame")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        frame_viewer_layout.addWidget(self.canvas)

        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(9)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.update_frame_display)
        slider_layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("0/9")
        slider_layout.addWidget(self.frame_label)

        frame_viewer_layout.addLayout(slider_layout)

        # Display options
        display_layout = QHBoxLayout()

        self.show_particles_check = QCheckBox("Show Particles")
        self.show_particles_check.setChecked(True)
        self.show_particles_check.stateChanged.connect(self.update_frame_display)
        display_layout.addWidget(self.show_particles_check)

        self.show_tracks_check = QCheckBox("Show Tracks")
        self.show_tracks_check.setChecked(True)
        self.show_tracks_check.stateChanged.connect(self.update_frame_display)
        display_layout.addWidget(self.show_tracks_check)

        self.animate_button = QPushButton("Animate")
        self.animate_button.clicked.connect(self.animate_tracks)
        display_layout.addWidget(self.animate_button)

        frame_viewer_layout.addLayout(display_layout)

        # Status message
        self.status_label = QLabel("Ready")
        right_layout.addWidget(self.frame_viewer_group)
        right_layout.addWidget(self.status_label)

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)

    def setup_connections(self):
        """Set up signal connections."""
        # Connect motion model combo box to motion parameters visibility
        self.motion_combo.currentIndexChanged.connect(self.update_motion_params)

        # Connect PSF model combo box to PSF parameters visibility
        self.psf_combo.currentIndexChanged.connect(self.update_psf_params)

        # Connect noise model combo box to noise parameters visibility
        self.noise_combo.currentIndexChanged.connect(self.update_noise_params)

        # Connect blinking checkbox to blinking parameters visibility
        self.blinking_check.stateChanged.connect(self.update_blinking_params)

        # Initialize visibility
        self.update_motion_params()
        self.update_psf_params()
        self.update_noise_params()
        self.update_blinking_params()

    def update_motion_params(self):
        """Update motion parameters visibility based on selected motion model."""
        motion_model = self.motion_combo.currentText()

        # Clear existing widgets
        for i in reversed(range(self.motion_params_layout.count())):
            item = self.motion_params_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.hide()
                    self.motion_params_layout.removeItem(item)

        # Add parameters for selected model
        if motion_model == "Brownian":
            self.motion_params_layout.addRow("Diffusion Coefficient:", self.diffusion_coef_spin)
            self.diffusion_coef_spin.show()
        elif motion_model == "Directed":
            self.motion_params_layout.addRow("Velocity:", self.velocity_spin)
            self.motion_params_layout.addRow("Direction Change Prob:", self.direction_change_spin)
            self.velocity_spin.show()
            self.direction_change_spin.show()
        elif motion_model == "Confined":
            self.motion_params_layout.addRow("Diffusion Coefficient:", self.diffusion_coef_spin)
            self.motion_params_layout.addRow("Confinement Strength:", self.confinement_strength_spin)
            self.motion_params_layout.addRow("Confinement Radius:", self.confinement_radius_spin)
            self.diffusion_coef_spin.show()
            self.confinement_strength_spin.show()
            self.confinement_radius_spin.show()

    def update_psf_params(self):
        """Update PSF parameters visibility based on selected PSF model."""
        psf_model = self.psf_combo.currentText()

        # Hide all parameters
        self.psf_sigma_spin.hide()
        self.airy_radius_spin.hide()

        # Show relevant parameters
        if psf_model == "Gaussian":
            for i in reversed(range(self.psf_group.layout().count())):
                item = self.psf_group.layout().itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.hide()

            self.psf_group.layout().addRow("PSF Model:", self.psf_combo)
            self.psf_group.layout().addRow("Sigma:", self.psf_sigma_spin)
            self.psf_combo.show()
            self.psf_sigma_spin.show()
        elif psf_model == "Airy Disk":
            for i in reversed(range(self.psf_group.layout().count())):
                item = self.psf_group.layout().itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.hide()

            self.psf_group.layout().addRow("PSF Model:", self.psf_combo)
            self.psf_group.layout().addRow("Airy Radius:", self.airy_radius_spin)
            self.psf_combo.show()
            self.airy_radius_spin.show()

    def update_noise_params(self):
        """Update noise parameters visibility based on selected noise model."""
        noise_model = self.noise_combo.currentText()

        # Hide all parameters
        self.read_noise_spin.hide()

        # Show relevant parameters
        if noise_model == "Poisson":
            for i in reversed(range(self.noise_group.layout().count())):
                item = self.noise_group.layout().itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.hide()

            self.noise_group.layout().addRow("Noise Model:", self.noise_combo)
            self.noise_group.layout().addRow("Signal-to-Noise Ratio:", self.snr_spin)
            self.noise_group.layout().addRow("Background Level:", self.background_spin)
            self.noise_combo.show()
            self.snr_spin.show()
            self.background_spin.show()
        elif noise_model == "Gaussian":
            for i in reversed(range(self.noise_group.layout().count())):
                item = self.noise_group.layout().itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.hide()

            self.noise_group.layout().addRow("Noise Model:", self.noise_combo)
            self.noise_group.layout().addRow("Signal-to-Noise Ratio:", self.snr_spin)
            self.noise_group.layout().addRow("Background Level:", self.background_spin)
            self.noise_group.layout().addRow("Read Noise:", self.read_noise_spin)
            self.noise_combo.show()
            self.snr_spin.show()
            self.background_spin.show()
            self.read_noise_spin.show()
        elif noise_model == "Poisson + Gaussian":
            for i in reversed(range(self.noise_group.layout().count())):
                item = self.noise_group.layout().itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.hide()

            self.noise_group.layout().addRow("Noise Model:", self.noise_combo)
            self.noise_group.layout().addRow("Signal-to-Noise Ratio:", self.snr_spin)
            self.noise_group.layout().addRow("Background Level:", self.background_spin)
            self.noise_group.layout().addRow("Read Noise:", self.read_noise_spin)
            self.noise_combo.show()
            self.snr_spin.show()
            self.background_spin.show()
            self.read_noise_spin.show()

    def update_blinking_params(self):
        """Update blinking parameters visibility based on blinking checkbox."""
        enable_blinking = self.blinking_check.isChecked()

        # Show/hide blinking parameters
        self.on_prob_spin.setVisible(enable_blinking)
        self.off_prob_spin.setVisible(enable_blinking)

        # Update layout labels
        for i in reversed(range(self.blinking_group.layout().count())):
            item = self.blinking_group.layout().itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None and widget != self.blinking_check:
                    widget.hide()

        self.blinking_group.layout().addRow(self.blinking_check)

        if enable_blinking:
            self.blinking_group.layout().addRow("On Probability:", self.on_prob_spin)
            self.blinking_group.layout().addRow("Off Probability:", self.off_prob_spin)
            self.on_prob_spin.show()
            self.off_prob_spin.show()

    def browse_output_path(self):
        """Open file dialog to select output path."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", os.path.expanduser("~")
        )

        if directory:
            self.output_path_edit.setText(directory)

    def run_simulation(self):
        """Run the simulation based on current parameters."""
        # Get parameters
        width = self.width_spin.value()
        height = self.height_spin.value()
        num_frames = self.num_frames_spin.value()
        num_particles = self.num_particles_spin.value()

        # Update status
        self.status_label.setText("Simulating...")
        QApplication.processEvents()

        # Reset particle generator
        self.particle_generator = ParticleGenerator(image_size=(height, width))

        try:
            # Create particles
            particles = self.particle_generator.create_random_particles(
                num_particles=num_particles,
                intensity_range=(0.5, 1.0),
                size_range=(0.8, 1.2)
            )

            # Apply motion
            motion_model = self.motion_combo.currentText()

            if motion_model == "Brownian":
                self.particle_generator.apply_brownian_motion(
                    particles=particles,
                    num_frames=num_frames,
                    diffusion_coefficient=self.diffusion_coef_spin.value()
                )
            elif motion_model == "Directed":
                self.particle_generator.apply_directed_motion(
                    particles=particles,
                    num_frames=num_frames,
                    velocity_range=(self.velocity_spin.value() * 0.8, self.velocity_spin.value() * 1.2),
                    direction_change_prob=self.direction_change_spin.value()
                )
            elif motion_model == "Confined":
                self.particle_generator.apply_confined_diffusion(
                    particles=particles,
                    num_frames=num_frames,
                    diffusion_coefficient=self.diffusion_coef_spin.value(),
                    confinement_strength=self.confinement_strength_spin.value(),
                    confinement_radius=self.confinement_radius_spin.value()
                )

            # Apply blinking if enabled
            if self.blinking_check.isChecked():
                blinking_states = self.particle_generator.apply_blinking(
                    particles=particles,
                    num_frames=num_frames,
                    on_probability=self.on_prob_spin.value(),
                    off_probability=self.off_prob_spin.value()
                )

            # Create PSF model
            psf_model_name = self.psf_combo.currentText()

            if psf_model_name == "Gaussian":
                psf_model = GaussianPSF(
                    image_size=(height, width),
                    sigma=self.psf_sigma_spin.value()
                )
            elif psf_model_name == "Airy Disk":
                psf_model = AiryDiskPSF(
                    image_size=(height, width),
                    airy_radius=self.airy_radius_spin.value()
                )

            # Create noise model
            noise_model_name = self.noise_combo.currentText()
            snr = self.snr_spin.value()
            background = self.background_spin.value()

            if noise_model_name == "Poisson":
                noise_model = PoissonNoise(scaling_factor=100.0)
            elif noise_model_name == "Gaussian":
                noise_model = GaussianNoise(sigma=10.0 / snr)
            elif noise_model_name == "Poisson + Gaussian":
                noise_model = MixedNoise(
                    photon_scaling=100.0,
                    read_noise=self.read_noise_spin.value(),
                    offset=0.0
                )

            # Generate frames
            frames = []
            positions = []
            track_ids = []

            for frame_idx in range(num_frames):
                # Get active particles for this frame
                active_particles = self.particle_generator.get_active_particles(frame_idx)

                # Extract positions and intensities
                particle_positions = []
                particle_intensities = []
                particle_sizes = []
                particle_ids = []

                for particle in active_particles:
                    pos = particle.get_position(frame_idx)
                    particle_positions.append(pos)
                    particle_intensities.append(particle.intensity)
                    particle_sizes.append(particle.size)
                    particle_ids.append(particle.id)

                # Generate PSF image
                if particle_positions:
                    psf_image = psf_model.generate(
                        positions=particle_positions,
                        intensities=particle_intensities,
                        sizes=particle_sizes
                    )
                else:
                    psf_image = np.zeros((height, width), dtype=np.float32)

                # Apply noise
                signal_max = np.max(psf_image) if np.max(psf_image) > 0 else 1.0
                bg_level = signal_max / snr if signal_max > 0 else background

                noisy_image = noise_model.apply(psf_image + bg_level) - bg_level

                # Clip to [0, 1] and add to frames
                noisy_image = np.clip(noisy_image, 0, 1)
                frames.append(noisy_image)

                # Create position target (binary mask with particle positions)
                pos_target = np.zeros((height, width), dtype=np.float32)
                for pos in particle_positions:
                    y, x = int(round(pos[0])), int(round(pos[1]))  # Changed for (y, x) order
                    if 0 <= y < height and 0 <= x < width:
                        pos_target[y, x] = 1.0

                positions.append(pos_target)

                # Create track ID target
                track_id_target = np.zeros((height, width), dtype=np.int32)
                for pos, pid in zip(particle_positions, particle_ids):
                    y, x = int(round(pos[0])), int(round(pos[1]))  # Changed for (y, x) order
                    if 0 <= y < height and 0 <= x < width:
                        track_id_target[y, x] = pid + 1  # Add 1 because 0 is background

                track_ids.append(track_id_target)

            # Convert to arrays
            self.frames = np.stack(frames)
            self.positions = np.stack(positions)
            self.track_ids = np.stack(track_ids)

            # Update frame slider
            self.frame_slider.setMaximum(num_frames - 1)
            self.frame_label.setText(f"0/{num_frames-1}")

            # Update frame display
            self.update_frame_display()

            # Update status
            self.status_label.setText("Simulation completed")

            # Save output if requested
            if self.save_check.isChecked():
                self.save_simulation()

            # Emit signal with simulation results
            self.simulation_done.emit({
                'frames': self.frames,
                'positions': self.positions,
                'track_ids': self.track_ids
            })

        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def update_frame_display(self):
        """Update the displayed frame."""
        if self.frames is None:
            return

        # Get current frame index
        frame_idx = self.frame_slider.value()
        self.frame_label.setText(f"{frame_idx}/{len(self.frames)-1}")

        # Clear axes
        self.ax.clear()

        # Display the frame - with origin='lower' to match coordinate system
        self.ax.imshow(self.frames[frame_idx], cmap='gray', origin='lower')

        # Overlay positions if requested
        if self.show_particles_check.isChecked():
            # Find particle positions in this frame
            active_particles = self.particle_generator.get_active_particles(frame_idx)

            if active_particles:
                positions = [p.get_position(frame_idx) for p in active_particles]
                # Using the correct coordinate order for plotting
                # Positions are stored as (y, x), but plotting needs x, y
                self.ax.scatter(
                    [p[1] for p in positions],  # x values (second element in (y, x))
                    [p[0] for p in positions],  # y values (first element in (y, x))
                    c='red', s=20, alpha=0.7, marker='o'
                )

        # Overlay tracks if requested
        if self.show_tracks_check.isChecked():
            # Get tracks up to current frame
            for particle in self.particle_generator.particles:
                # Check if track exists at current frame
                if frame_idx < len(particle.positions):
                    # Get track history
                    history = particle.positions[:frame_idx+1]

                    # Plot track history - note the coordinate order
                    self.ax.plot(
                        [p[1] for p in history],  # x values (from position[1])
                        [p[0] for p in history],  # y values (from position[0])
                        '-', color='blue', alpha=0.5
                    )

        # Set title and remove ticks
        self.ax.set_title(f"Frame {frame_idx}")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Update canvas
        self.canvas.draw()

    def animate_tracks(self):
        """Create and display an animation of the tracks."""
        if self.frames is None:
            return

        try:
            # Create tracks in the required format
            tracks = []

            for particle in self.particle_generator.particles:
                # Convert particle positions from (y, x) to proper numpy array
                # For visualization, we need to swap to (x, y) for the animator
                track_array = np.array([(pos[1], pos[0]) for pos in particle.positions])
                tracks.append(track_array)

            # Create animator
            animator = TrackingAnimator(self.frames, tracks)

            # Create animation with origin='lower'
            anim = animator.create_animation(
                figsize=(8, 6),
                cmap='gray',
                line_alpha=0.7,
                point_size=20,
                interval=200,
                history_length=5
            )

            # Show animation in a new window
            plt.show()

        except Exception as e:
            logger.error(f"Error creating animation: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def save_simulation(self):
        """Save the simulation results to disk."""
        if self.frames is None:
            return

        # Get output path
        output_path = self.output_path_edit.text()

        if not output_path:
            # Ask for output path if not set
            output_path = QFileDialog.getExistingDirectory(
                self, "Select Output Directory", os.path.expanduser("~")
            )

            if not output_path:
                return

            self.output_path_edit.setText(output_path)

        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save frames as TIFF stack
            frames_path = os.path.join(output_path, f"frames_{timestamp}.tif")
            tifffile.imwrite(frames_path, (self.frames * 255).astype(np.uint8))

            # Save positions and track IDs
            positions_path = os.path.join(output_path, f"positions_{timestamp}.npy")
            np.save(positions_path, self.positions)

            track_ids_path = os.path.join(output_path, f"track_ids_{timestamp}.npy")
            np.save(track_ids_path, self.track_ids)

            # Save parameters as JSON
            params = {
                'width': self.width_spin.value(),
                'height': self.height_spin.value(),
                'num_frames': self.num_frames_spin.value(),
                'num_particles': self.num_particles_spin.value(),
                'motion_model': self.motion_combo.currentText(),
                'psf_model': self.psf_combo.currentText(),
                'noise_model': self.noise_combo.currentText(),
                'snr': self.snr_spin.value(),
                'background': self.background_spin.value(),
                'timestamp': timestamp
            }

            params_path = os.path.join(output_path, f"params_{timestamp}.json")
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=4)

            self.status_label.setText(f"Saved simulation to {output_path}")
            logger.info(f"Saved simulation to {output_path}")

        except Exception as e:
            logger.error(f"Error saving simulation: {str(e)}")
            self.status_label.setText(f"Error saving: {str(e)}")

    def clear_simulation(self):
        """Clear the current simulation."""
        self.frames = None
        self.positions = None
        self.track_ids = None

        # Clear axes
        self.ax.clear()
        self.ax.set_title("Simulated Frame")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Update canvas
        self.canvas.draw()

        # Reset status
        self.status_label.setText("Ready")

        # Reset frame slider
        self.frame_slider.setValue(0)
        self.frame_slider.setMaximum(9)
        self.frame_label.setText("0/9")


class TrainingTab(QWidget):
    """Tab for training particle tracking models."""

    def __init__(self, parent=None):
        """Initialize the training tab."""
        super().__init__(parent)

        # Initialize training manager
        self.training_manager = TrainingManager()

        # Set up UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QHBoxLayout(self)

        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Data group
        data_group = QGroupBox("Data Configuration")
        data_layout = QFormLayout(data_group)

        # Data source
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Simulated", "Load from File"])
        data_layout.addRow("Data Source:", self.data_source_combo)

        # Simulated data parameters
        self.sim_data_group = QGroupBox("Simulated Data Parameters")
        sim_data_layout = QFormLayout(self.sim_data_group)

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(100, 10000)
        self.num_samples_spin.setValue(1000)
        sim_data_layout.addRow("Number of Samples:", self.num_samples_spin)

        self.frame_sequence_spin = QSpinBox()
        self.frame_sequence_spin.setRange(1, 20)
        self.frame_sequence_spin.setValue(5)
        sim_data_layout.addRow("Frames per Sequence:", self.frame_sequence_spin)

        self.particles_per_frame_min = QSpinBox()
        self.particles_per_frame_min.setRange(1, 100)
        self.particles_per_frame_min.setValue(5)

        self.particles_per_frame_max = QSpinBox()
        self.particles_per_frame_max.setRange(1, 100)
        self.particles_per_frame_max.setValue(30)

        particles_layout = QHBoxLayout()
        particles_layout.addWidget(QLabel("Min:"))
        particles_layout.addWidget(self.particles_per_frame_min)
        particles_layout.addWidget(QLabel("Max:"))
        particles_layout.addWidget(self.particles_per_frame_max)

        sim_data_layout.addRow("Particles per Frame:", particles_layout)

        motion_layout = QHBoxLayout()
        self.motion_brownian_radio = QRadioButton("Brownian")
        self.motion_directed_radio = QRadioButton("Directed")
        self.motion_confined_radio = QRadioButton("Confined")

        motion_group = QButtonGroup(self)
        motion_group.addButton(self.motion_brownian_radio)
        motion_group.addButton(self.motion_directed_radio)
        motion_group.addButton(self.motion_confined_radio)

        self.motion_brownian_radio.setChecked(True)

        motion_layout.addWidget(self.motion_brownian_radio)
        motion_layout.addWidget(self.motion_directed_radio)
        motion_layout.addWidget(self.motion_confined_radio)

        sim_data_layout.addRow("Motion Model:", motion_layout)

        # Data loading parameters
        self.load_data_group = QGroupBox("Load Data Parameters")
        load_data_layout = QFormLayout(self.load_data_group)

        self.train_data_path_edit = QLineEdit()
        self.train_data_path_edit.setReadOnly(True)

        train_browse_button = QPushButton("Browse...")
        train_browse_button.clicked.connect(lambda: self.browse_data_path(self.train_data_path_edit, "Train"))

        train_path_layout = QHBoxLayout()
        train_path_layout.addWidget(self.train_data_path_edit)
        train_path_layout.addWidget(train_browse_button)

        load_data_layout.addRow("Training Data:", train_path_layout)

        self.val_data_path_edit = QLineEdit()
        self.val_data_path_edit.setReadOnly(True)

        val_browse_button = QPushButton("Browse...")
        val_browse_button.clicked.connect(lambda: self.browse_data_path(self.val_data_path_edit, "Validation"))

        val_path_layout = QHBoxLayout()
        val_path_layout.addWidget(self.val_data_path_edit)
        val_path_layout.addWidget(val_browse_button)

        load_data_layout.addRow("Validation Data:", val_path_layout)

        # Model parameters
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout(model_group)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Simple", "Dual Branch", "Attentive"])
        model_layout.addRow("Model Type:", self.model_type_combo)

        self.model_depth_spin = QSpinBox()
        self.model_depth_spin.setRange(2, 6)
        self.model_depth_spin.setValue(4)
        model_layout.addRow("Model Depth:", self.model_depth_spin)

        self.base_filters_spin = QSpinBox()
        self.base_filters_spin.setRange(16, 128)
        self.base_filters_spin.setValue(64)
        self.base_filters_spin.setSingleStep(16)
        model_layout.addRow("Base Filters:", self.base_filters_spin)

        # Training parameters
        training_group = QGroupBox("Training Configuration")
        training_layout = QFormLayout(training_group)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        training_layout.addRow("Batch Size:", self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        training_layout.addRow("Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        training_layout.addRow("Learning Rate:", self.lr_spin)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "AdamW"])
        training_layout.addRow("Optimizer:", self.optimizer_combo)

        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["None", "Plateau", "Cosine", "Step"])
        training_layout.addRow("Scheduler:", self.scheduler_combo)

        # Training ID
        self.training_id_edit = QLineEdit()
        self.training_id_edit.setText(f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        training_layout.addRow("Training ID:", self.training_id_edit)

        # Control buttons
        control_layout = QHBoxLayout()

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_button)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        # Add to left panel
        left_layout.addWidget(data_group)
        left_layout.addWidget(self.sim_data_group)
        left_layout.addWidget(self.load_data_group)
        left_layout.addWidget(model_group)
        left_layout.addWidget(training_group)
        left_layout.addLayout(control_layout)
        left_layout.addStretch()

        # Right panel (visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Training progress group
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)

        # Matplotlib canvas for training metrics
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = MatplotlibCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        progress_layout.addWidget(self.canvas)

        # Active trainings group
        trainings_group = QGroupBox("Active Trainings")
        trainings_layout = QVBoxLayout(trainings_group)

        self.trainings_list = QListWidget()
        self.trainings_list.itemSelectionChanged.connect(self.update_selected_training)
        trainings_layout.addWidget(self.trainings_list)

        trainings_buttons_layout = QHBoxLayout()

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_trainings)
        trainings_buttons_layout.addWidget(self.refresh_button)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_selected_model)
        self.load_model_button.setEnabled(False)
        trainings_buttons_layout.addWidget(self.load_model_button)

        trainings_layout.addLayout(trainings_buttons_layout)

        # Add to right panel
        right_layout.addWidget(progress_group)
        right_layout.addWidget(trainings_group)

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])

        main_layout.addWidget(splitter)

        # Connect data source combo box
        self.data_source_combo.currentIndexChanged.connect(self.update_data_source)

        # Initialize data source visibility
        self.update_data_source()

        # Timer for updating training progress
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_training_progress)
        self.update_timer.start(1000)  # 1 second interval

    def update_data_source(self):
        """Update data parameters visibility based on selected data source."""
        data_source = self.data_source_combo.currentText()

        if data_source == "Simulated":
            self.sim_data_group.show()
            self.load_data_group.hide()
        else:
            self.sim_data_group.hide()
            self.load_data_group.show()

    def browse_data_path(self, line_edit, data_type):
        """
        Open file dialog to select data path.

        Args:
            line_edit: QLineEdit to update with selected path
            data_type: String indicating data type ("Train" or "Validation")
        """
        directory = QFileDialog.getExistingDirectory(
            self, f"Select {data_type} Data Directory", os.path.expanduser("~")
        )

        if directory:
            line_edit.setText(directory)

    def start_training(self):
        """Start the training process."""
        # Get training ID
        training_id = self.training_id_edit.text()

        if not training_id:
            QMessageBox.warning(self, "Warning", "Please enter a training ID")
            return

        # Update UI
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Preparing for training...")
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        try:
            # Create data loaders
            data_source = self.data_source_combo.currentText()

            if data_source == "Simulated":
                # Get simulation parameters
                num_samples = self.num_samples_spin.value()
                frame_sequence_length = self.frame_sequence_spin.value()
                particles_min = self.particles_per_frame_min.value()
                particles_max = self.particles_per_frame_max.value()

                # Determine motion model
                if self.motion_brownian_radio.isChecked():
                    motion_model = "brownian"
                elif self.motion_directed_radio.isChecked():
                    motion_model = "directed"
                else:
                    motion_model = "confined"

                # Create simulated dataset
                self.status_label.setText("Creating simulated dataset...")
                QApplication.processEvents()

                train_dataset = SimulatedParticleDataset(
                    num_samples=num_samples,
                    frame_sequence_length=frame_sequence_length,
                    particles_per_frame=(particles_min, particles_max),
                    motion_model=motion_model
                )

                # Create a smaller validation set
                val_dataset = SimulatedParticleDataset(
                    num_samples=max(100, num_samples // 10),
                    frame_sequence_length=frame_sequence_length,
                    particles_per_frame=(particles_min, particles_max),
                    motion_model=motion_model
                )

                # Create data loaders
                batch_size = self.batch_size_spin.value()

                from torch.utils.data import DataLoader

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

            else:
                # Load data from file
                train_data_path = self.train_data_path_edit.text()
                val_data_path = self.val_data_path_edit.text()

                if not train_data_path:
                    QMessageBox.warning(self, "Warning", "Please select training data path")
                    self.train_button.setEnabled(True)
                    self.stop_button.setEnabled(False)
                    return

                if not val_data_path:
                    val_data_path = None

                # Create data loaders
                self.status_label.setText("Loading data...")
                QApplication.processEvents()

                loaders = create_dataloaders(
                    train_data_source=train_data_path,
                    val_data_source=val_data_path,
                    batch_size=self.batch_size_spin.value(),
                    frame_sequence_length=self.frame_sequence_spin.value()
                )

                train_loader = loaders['train']
                val_loader = loaders['val']

            # Set up model configuration
            model_type = self.model_type_combo.currentText().lower().replace(" ", "_")

            model_config = {
                'input_channels': 1,
                'num_frames': self.frame_sequence_spin.value(),
                'base_filters': self.base_filters_spin.value(),
                'depth': self.model_depth_spin.value()
            }

            # Set up optimizer
            optimizer_type = self.optimizer_combo.currentText().lower()

            # Set up scheduler
            scheduler_type = self.scheduler_combo.currentText().lower()
            if scheduler_type == "none":
                scheduler_type = None

            # Set up training parameters
            epochs = self.epochs_spin.value()
            lr = self.lr_spin.value()

            # Clear current plot
            self.ax.clear()
            self.ax.set_title("Training Metrics")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss")
            self.ax.grid(True)
            self.canvas.draw()

            # Start training
            self.status_label.setText("Starting training...")
            QApplication.processEvents()

            # Add training to list
            self.add_training_to_list(training_id)

            # Start training in background
            def training_callback(training_id, result):
                self.train_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.status_label.setText(f"Training {training_id} completed")
                self.progress_bar.setValue(100)

            self.training_manager.start_training(
                training_id=training_id,
                train_loader=train_loader,
                val_loader=val_loader,
                model_type=model_type,
                model_config=model_config,
                optimizer_type=optimizer_type,
                lr=lr,
                epochs=epochs,
                callback=training_callback
            )

        except Exception as e:
            logger.error(f"Error starting training: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def stop_training(self):
        """Stop the current training process."""
        # Get training ID
        training_id = self.training_id_edit.text()

        if not training_id:
            return

        # Stop training
        self.training_manager.stop_training(training_id)

        # Update UI
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(f"Training {training_id} stopped")

    def update_training_progress(self):
        """Update the training progress display with robust error handling."""
        # Get training ID
        training_id = self.training_id_edit.text()

        if not training_id:
            return

        # Get training status
        status = self.training_manager.get_training_status(training_id)

        # Update progress
        if status['task_status'] == 'running':
            if 'current_epoch' in status and 'history' in status:
                current_epoch = status['current_epoch']
                epochs = self.epochs_spin.value()

                # Update progress bar
                progress = min(100, int(100 * current_epoch / epochs))
                self.progress_bar.setValue(progress)

                # Update status
                self.status_label.setText(f"Training {training_id}: Epoch {current_epoch}/{epochs}")

                # Update plot
                try:
                    history = status['history']

                    if 'train_loss' in history and history['train_loss']:
                        train_loss = history['train_loss']

                        # Clear axis
                        self.ax.clear()

                        # Plot training loss
                        epochs_range = range(1, len(train_loss) + 1)
                        self.ax.plot(epochs_range, train_loss, 'b-', label='Train Loss')

                        # Plot validation loss only if it exists and has data
                        if 'val_loss' in history and history['val_loss'] and len(history['val_loss']) > 0:
                            val_loss = history['val_loss']

                            # Create appropriate validation x-axis points
                            if len(val_loss) == len(train_loss):
                                # Same frequency - use same x values
                                val_epochs = epochs_range
                            else:
                                # Different frequency - create appropriate scale
                                # This assumes validation happens at regular intervals
                                val_epochs = np.linspace(1, len(train_loss), len(val_loss))

                            self.ax.plot(val_epochs, val_loss, 'r-', label='Validation Loss')

                        # Set title and labels
                        self.ax.set_title("Training Metrics")
                        self.ax.set_xlabel("Epoch")
                        self.ax.set_ylabel("Loss")
                        self.ax.legend()
                        self.ax.grid(True)

                        # Update canvas
                        self.canvas.draw()
                except Exception as e:
                    # Handle plotting errors gracefully
                    logger.error(f"Error updating training plot: {str(e)}")
                    # Don't let plotting errors crash the progress update
                    pass

        # Refresh trainings list
        self.refresh_trainings()

    def add_training_to_list(self, training_id):
        """
        Add a training to the list of active trainings.

        Args:
            training_id: ID of the training to add
        """
        # Check if already in list
        for i in range(self.trainings_list.count()):
            if self.trainings_list.item(i).text() == training_id:
                return

        # Add to list
        self.trainings_list.addItem(training_id)

    def refresh_trainings(self):
        """Refresh the list of active trainings."""
        # Get all trainings
        trainings = self.training_manager.list_trainings()

        # Clear list
        self.trainings_list.clear()

        # Add trainings to list
        for training in trainings:
            item = QListWidgetItem(training['training_id'])

            # Set color based on status
            if training['task_status'] == 'running':
                item.setForeground(QColor(0, 128, 0))  # Green
            elif training['task_status'] == 'failed':
                item.setForeground(QColor(255, 0, 0))  # Red
            elif training['task_status'] == 'completed':
                item.setForeground(QColor(0, 0, 255))  # Blue

            self.trainings_list.addItem(item)

    def update_selected_training(self):
        """Update UI based on selected training."""
        # Get selected training
        selected_items = self.trainings_list.selectedItems()

        if not selected_items:
            self.load_model_button.setEnabled(False)
            return

        # Enable load model button
        self.load_model_button.setEnabled(True)

        # Get training ID
        training_id = selected_items[0].text()

        # Get training status
        status = self.training_manager.get_training_status(training_id)

        # Update plot if history available
        if 'history' in status:
            try:
                history = status['history']

                if 'train_loss' in history and len(history['train_loss']) > 0:
                    train_loss = history['train_loss']

                    # Clear axis
                    self.ax.clear()

                    # Plot training loss
                    epochs_range = range(1, len(train_loss) + 1)
                    self.ax.plot(epochs_range, train_loss, 'b-', label='Train Loss')

                    # Plot validation loss only if it exists and has data
                    if 'val_loss' in history and len(history['val_loss']) > 0:
                        val_loss = history['val_loss']

                        # Make sure lengths match
                        if len(val_loss) == len(epochs_range):
                            self.ax.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
                        else:
                            # Create appropriate validation x-axis points
                            val_epochs = range(1, len(val_loss) + 1)
                            self.ax.plot(val_epochs, val_loss, 'r-', label='Validation Loss')

                    # Set title and labels
                    self.ax.set_title(f"Training Metrics - {training_id}")
                    self.ax.set_xlabel("Epoch")
                    self.ax.set_ylabel("Loss")
                    self.ax.legend()
                    self.ax.grid(True)

                    # Update canvas
                    self.canvas.draw()
            except Exception as e:
                # Gracefully handle plotting errors
                logger.error(f"Error plotting training history: {str(e)}")

    def load_selected_model(self):
        """Load the selected model for prediction."""
        # Get selected training
        selected_items = self.trainings_list.selectedItems()

        if not selected_items:
            return

        # Get training ID
        training_id = selected_items[0].text()

        # Emit signal or notify parent to load model
        # This depends on how you want to handle communication between tabs
        logger.info(f"Loading model for training {training_id}")

        # For now, just show a message
        QMessageBox.information(
            self, "Load Model", f"Model for training {training_id} selected"
        )


class PredictionTab(QWidget):
    """Tab for making predictions with trained models."""

    def __init__(self, parent=None):
        """Initialize the prediction tab."""
        super().__init__(parent)

        # Initialize prediction manager
        self.prediction_manager = PredictionManager()

        # Set up UI
        self.setup_ui()

        # Initialize prediction data
        self.frames = None
        self.current_prediction = None

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QHBoxLayout(self)

        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Data group
        data_group = QGroupBox("Input Data")
        data_layout = QFormLayout(data_group)

        # Data source
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Load from File", "Use Simulation"])
        data_layout.addRow("Data Source:", self.data_source_combo)

        # File path
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_data_file)

        file_path_layout = QHBoxLayout()
        file_path_layout.addWidget(self.file_path_edit)
        file_path_layout.addWidget(browse_button)

        data_layout.addRow("File Path:", file_path_layout)

        # Model group
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        # Model path
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)

        model_browse_button = QPushButton("Browse...")
        model_browse_button.clicked.connect(self.browse_model_file)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_browse_button)

        model_layout.addRow("Model Path:", model_path_layout)

        # Model type
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Simple", "Dual Branch", "Attentive"])
        model_layout.addRow("Model Type:", self.model_type_combo)

        # Detection parameters
        detection_group = QGroupBox("Detection Parameters")
        detection_layout = QFormLayout(detection_group)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.05)
        detection_layout.addRow("Detection Threshold:", self.threshold_spin)

        self.nms_radius_spin = QSpinBox()
        self.nms_radius_spin.setRange(1, 10)
        self.nms_radius_spin.setValue(3)
        detection_layout.addRow("NMS Radius:", self.nms_radius_spin)

        # Tracking parameters
        tracking_group = QGroupBox("Tracking Parameters")
        tracking_layout = QFormLayout(tracking_group)

        self.link_particles_check = QCheckBox()
        self.link_particles_check.setChecked(True)
        tracking_layout.addRow("Link Particles:", self.link_particles_check)

        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(1.0, 50.0)
        self.max_distance_spin.setValue(20.0)
        self.max_distance_spin.setSingleStep(1.0)
        tracking_layout.addRow("Max Link Distance:", self.max_distance_spin)

        # Prediction buttons
        predict_buttons_layout = QHBoxLayout()

        self.predict_button = QPushButton("Run Prediction")
        self.predict_button.clicked.connect(self.run_prediction)
        predict_buttons_layout.addWidget(self.predict_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_prediction)
        predict_buttons_layout.addWidget(self.clear_button)

        # Add to left panel
        left_layout.addWidget(data_group)
        left_layout.addWidget(model_group)
        left_layout.addWidget(detection_group)
        left_layout.addWidget(tracking_group)
        left_layout.addLayout(predict_buttons_layout)
        left_layout.addStretch()

        # Right panel (visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Frame viewer
        self.frame_viewer_group = QGroupBox("Prediction Viewer")
        frame_viewer_layout = QVBoxLayout(self.frame_viewer_group)

        # Matplotlib canvas for frame display
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.canvas = MatplotlibCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Prediction")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        frame_viewer_layout.addWidget(self.canvas)

        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.update_frame_display)
        slider_layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("0/0")
        slider_layout.addWidget(self.frame_label)

        frame_viewer_layout.addLayout(slider_layout)

        # Display options
        display_layout = QHBoxLayout()

        self.show_positions_check = QCheckBox("Show Positions")
        self.show_positions_check.setChecked(True)
        self.show_positions_check.stateChanged.connect(self.update_frame_display)
        display_layout.addWidget(self.show_positions_check)

        self.show_tracks_check = QCheckBox("Show Tracks")
        self.show_tracks_check.setChecked(True)
        self.show_tracks_check.stateChanged.connect(self.update_frame_display)
        display_layout.addWidget(self.show_tracks_check)

        self.show_probability_check = QCheckBox("Show Probability Map")
        self.show_probability_check.setChecked(False)
        self.show_probability_check.stateChanged.connect(self.update_frame_display)
        display_layout.addWidget(self.show_probability_check)

        frame_viewer_layout.addLayout(display_layout)

        # Animation/save buttons
        action_layout = QHBoxLayout()

        self.animate_button = QPushButton("Animate")
        self.animate_button.clicked.connect(self.animate_prediction)
        self.animate_button.setEnabled(False)
        action_layout.addWidget(self.animate_button)

        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_prediction)
        self.save_button.setEnabled(False)
        action_layout.addWidget(self.save_button)

        frame_viewer_layout.addLayout(action_layout)

        # Status message
        self.status_label = QLabel("Ready")
        right_layout.addWidget(self.frame_viewer_group)
        right_layout.addWidget(self.status_label)

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)

        # Timer for updating prediction progress
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_prediction_progress)
        self.update_timer.start(1000)  # 1 second interval

    def browse_data_file(self):
        """Open file dialog to select data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", os.path.expanduser("~"),
            "Image Files (*.tif *.tiff *.h5 *.hdf5);;All Files (*)"
        )

        if file_path:
            self.file_path_edit.setText(file_path)

            # Try to load the file
            try:
                self.load_data_file(file_path)
            except Exception as e:
                logger.error(f"Error loading data file: {str(e)}")
                self.status_label.setText(f"Error loading file: {str(e)}")

    def browse_model_file(self):
        """Open file dialog to select model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", os.path.expanduser("~"),
            "PyTorch Models (*.pth);;All Files (*)"
        )

        if file_path:
            self.model_path_edit.setText(file_path)

    def load_data_file(self, file_path):
        """
        Load data from file.

        Args:
            file_path: Path to the data file
        """
        self.status_label.setText(f"Loading {file_path}...")
        QApplication.processEvents()

        # Check file extension
        _, ext = os.path.splitext(file_path)

        if ext.lower() in ['.tif', '.tiff']:
            # Load TIFF file
            self.frames = tifffile.imread(file_path)

            # Ensure 3D array
            if self.frames.ndim == 2:
                self.frames = self.frames[np.newaxis, ...]

        elif ext.lower() in ['.h5', '.hdf5']:
            # Load HDF5 file
            import h5py

            with h5py.File(file_path, 'r') as f:
                # Check for frames dataset
                if 'frames' in f:
                    self.frames = np.array(f['frames'])
                else:
                    # Try to find other datasets
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 2:
                            self.frames = np.array(f[key])
                            break

                    if self.frames is None:
                        raise ValueError("No suitable dataset found in HDF5 file")

        else:
            # Try to load as image
            try:
                self.frames = np.array(Image.open(file_path))

                # Ensure 3D array
                if self.frames.ndim == 2:
                    self.frames = self.frames[np.newaxis, ...]

            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")

        # Normalize frames to [0, 1]
        if self.frames.max() > 1.0:
            self.frames = self.frames.astype(np.float32) / 255.0

        # Update frame slider
        self.frame_slider.setMaximum(len(self.frames) - 1)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"0/{len(self.frames)-1}")

        # Update frame display
        self.update_frame_display()

        self.status_label.setText(f"Loaded {len(self.frames)} frames")

    def update_frame_display(self):
        """Update the displayed frame."""
        if self.frames is None:
            return

        # Get current frame index
        frame_idx = self.frame_slider.value()
        self.frame_label.setText(f"{frame_idx}/{len(self.frames)-1}")

        # Clear axes
        self.ax.clear()

        # Display the frame - with origin='lower' to match coordinate system
        self.ax.imshow(self.frames[frame_idx], cmap='gray', origin='lower')

        # Check if prediction is available
        if self.current_prediction is not None:
            # Overlay positions if requested
            if self.show_positions_check.isChecked() and 'positions' in self.current_prediction:
                positions = self.current_prediction['positions']

                if frame_idx < len(positions):
                    pos = positions[frame_idx]
                    # Positions are stored as (y, x) order but plotting needs x, y
                    self.ax.scatter(
                        pos[:, 1], pos[:, 0],  # Switched to match (y, x) storage order
                        c='red', s=20, alpha=0.7, marker='o'
                    )

            # Overlay tracks if requested
            if self.show_tracks_check.isChecked() and 'tracks' in self.current_prediction and self.current_prediction['tracks'] is not None:
                tracks = self.current_prediction['tracks']

                # Generate colors for tracks
                track_colors = plt.cm.jet(np.linspace(0, 1, len(tracks)))

                for track_idx, track in enumerate(tracks):
                    # Check if track exists at current frame
                    if frame_idx < len(track) and not np.isnan(track[frame_idx, 0]):
                        # Get track history
                        history = track[:frame_idx+1]
                        valid_mask = ~np.isnan(history[:, 0])
                        history = history[valid_mask]

                        if len(history) > 0:
                            color = track_colors[track_idx]

                            # Plot track history - note the coordinate order could be switched
                            # based on how tracks are stored
                            if len(history) > 1:
                                # Switch coordinates for plotting if tracks are stored as (y, x)
                                self.ax.plot(
                                    history[:, 1], history[:, 0],  # Switched for (y, x) order
                                    '-', color=color, alpha=0.5
                                )

                            # Plot current position
                            self.ax.plot(
                                track[frame_idx, 1], track[frame_idx, 0],  # Switched for (y, x) order
                                'o', color=color, markersize=7
                            )

            # Overlay probability map if requested
            if (self.show_probability_check.isChecked() and
                ((('probability_maps' in self.current_prediction and
                   frame_idx < len(self.current_prediction['probability_maps'])) or
                 ('probability_map' in self.current_prediction)))):

                if 'probability_maps' in self.current_prediction:
                    prob_map = self.current_prediction['probability_maps'][frame_idx]
                else:
                    prob_map = self.current_prediction['probability_map']

                self.ax.imshow(prob_map, cmap='hot', alpha=0.5, origin='lower')

        # Set title and remove ticks
        self.ax.set_title(f"Frame {frame_idx}")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Update canvas
        self.canvas.draw()

    def run_prediction(self):
        """Run prediction on the loaded data."""
        # Check if data is loaded
        if self.frames is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return

        # Check if model path is set
        model_path = self.model_path_edit.text()
        if not model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file")
            return

        # Update UI
        self.predict_button.setEnabled(False)
        self.status_label.setText("Running prediction...")
        QApplication.processEvents()

        try:
            # Get model parameters
            model_type = self.model_type_combo.currentText().lower().replace(" ", "_")
            threshold = self.threshold_spin.value()
            nms_radius = self.nms_radius_spin.value()

            # Create predictor
            model_id = os.path.basename(model_path)
            predictor = self.prediction_manager.create_predictor(
                model_id=model_id,
                model_path=model_path,
                model_type=model_type,
                threshold=threshold,
                nms_radius=nms_radius
            )

            # Get tracking parameters
            link_particles = self.link_particles_check.isChecked()
            max_distance = self.max_distance_spin.value()

            # Run prediction
            prediction_id = f"pred_{os.path.basename(model_path)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            def prediction_callback(prediction_id, result):
                self.current_prediction = result
                self.predict_button.setEnabled(True)
                self.animate_button.setEnabled(True)
                self.save_button.setEnabled(True)
                self.status_label.setText(f"Prediction completed: {prediction_id}")
                self.update_frame_display()

            # Run prediction in background
            self.prediction_manager.predict_sequence(
                model_id=model_id,
                frames=self.frames,
                prediction_id=prediction_id,
                link_particles=link_particles,
                max_distance=max_distance,
                return_probability_maps=True,
                save_results=False,
                callback=prediction_callback
            )

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.predict_button.setEnabled(True)

    def update_prediction_progress(self):
        """Update the prediction progress."""
        # Check if prediction is running
        if not self.predict_button.isEnabled():
            # Update status
            self.status_label.setText("Running prediction...")

    def animate_prediction(self):
        """Create and display an animation of the prediction."""
        if self.frames is None or self.current_prediction is None:
            return

        try:
            # Check if tracks are available
            if 'tracks' not in self.current_prediction or self.current_prediction['tracks'] is None:
                QMessageBox.warning(self, "Warning", "No tracks available for animation")
                return

            # Create animation
            tracks = self.current_prediction['tracks']

            # Convert tracks if needed - if tracks are in (y, x) format, convert to (x, y) for animator
            # This depends on how tracks are stored in current_prediction
            converted_tracks = []
            for track in tracks:
                # Swap columns if tracks are stored as (y, x)
                converted_track = np.zeros_like(track)
                converted_track[:, 0] = track[:, 1]  # x = original y
                converted_track[:, 1] = track[:, 0]  # y = original x
                converted_tracks.append(converted_track)

            # Create animator
            animator = TrackingAnimator(self.frames, converted_tracks)

            # Create animation with origin='lower'
            anim = animator.create_animation(
                figsize=(8, 6),
                cmap='gray',
                line_alpha=0.7,
                point_size=20,
                interval=200,
                history_length=5
            )

            # Show animation in a new window
            plt.show()

        except Exception as e:
            logger.error(f"Error creating animation: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def save_prediction(self):
        """Save the prediction results to disk."""
        if self.frames is None or self.current_prediction is None:
            return

        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", os.path.expanduser("~")
        )

        if not output_dir:
            return

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save positions
            if 'positions' in self.current_prediction:
                positions_path = os.path.join(output_dir, f"positions_{timestamp}.npy")
                np.save(positions_path, self.current_prediction['positions'])

            # Save tracks
            if 'tracks' in self.current_prediction and self.current_prediction['tracks'] is not None:
                tracks_path = os.path.join(output_dir, f"tracks_{timestamp}.npy")
                np.save(tracks_path, self.current_prediction['tracks'])

            # Save probability maps
            if 'probability_maps' in self.current_prediction:
                prob_maps_path = os.path.join(output_dir, f"probability_maps_{timestamp}.npy")
                np.save(prob_maps_path, self.current_prediction['probability_maps'])

            # Save frames as TIFF stack
            frames_path = os.path.join(output_dir, f"frames_{timestamp}.tif")
            tifffile.imwrite(frames_path, (self.frames * 255).astype(np.uint8))

            # Generate visualization frames
            vis_dir = os.path.join(output_dir, f"visualization_{timestamp}")
            os.makedirs(vis_dir, exist_ok=True)

            # Create animator
            if 'tracks' in self.current_prediction and self.current_prediction['tracks'] is not None:
                # Convert tracks for visualization
                tracks = self.current_prediction['tracks']
                converted_tracks = []
                for track in tracks:
                    # Swap columns if tracks are stored as (y, x)
                    converted_track = np.zeros_like(track)
                    converted_track[:, 0] = track[:, 1]  # x = original y
                    converted_track[:, 1] = track[:, 0]  # y = original x
                    converted_tracks.append(converted_track)

                animator = TrackingAnimator(self.frames, converted_tracks)

                # Save frames
                animator.save_animation_frames(
                    output_dir=vis_dir,
                    format='png',
                    dpi=150
                )

            self.status_label.setText(f"Results saved to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            self.status_label.setText(f"Error saving: {str(e)}")

    def clear_prediction(self):
        """Clear the current prediction."""
        self.current_prediction = None
        self.animate_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # Update display
        self.update_frame_display()

        self.status_label.setText("Ready")


class MainWindow(QMainWindow):
    """Main window for Deep Particle Tracker."""

    def __init__(self, enable_console=True):
        """Initialize the main window."""
        super().__init__()

        # Set window properties
        self.setWindowTitle("Deep Particle Tracker")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create tabs
        self.simulation_tab = SimulationTab()
        self.training_tab = TrainingTab()
        self.prediction_tab = PredictionTab()

        # Add tabs to tab widget
        self.tab_widget.addTab(self.simulation_tab, "Simulation")
        self.tab_widget.addTab(self.training_tab, "Training")
        self.tab_widget.addTab(self.prediction_tab, "Prediction")

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)

        # Create console for logging if enabled
        if enable_console:
            self.console = QTextEdit()
            self.console.setReadOnly(True)
            self.console.setMaximumHeight(150)
            main_layout.addWidget(self.console)

            # Set up logging to console
            self.setup_logging()
        else:
            self.console = None

        # Connect signals between tabs
        self.connect_signals()

    def setup_logging(self):
        """Set up logging to console."""
        # Create console handler
        console_handler = ConsoleLogHandler(self.console)
        console_handler.setLevel(logging.INFO)

        # Set formatter
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)

    def connect_signals(self):
        """Connect signals between tabs."""
        # Connect simulation tab to training tab
        self.simulation_tab.simulation_done.connect(self.simulation_to_training)

    def simulation_to_training(self, data):
        """
        Handle simulation data for training.

        Args:
            data: Dictionary with simulation data
        """
        # Log the event
        logger.info("Simulation data ready for training")

        # Switch to training tab
        self.tab_widget.setCurrentWidget(self.training_tab)

        # TODO: Set up training tab with simulation data

    def closeEvent(self, event):
        """
        Handle application close event.

        Args:
            event: Close event
        """
        # Clean up resources
        thread_manager.shutdown(wait=True)
        event.accept()
