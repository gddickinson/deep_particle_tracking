"""
Visualization components for particle tracking data and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Optional, Union, Callable
import torch
import io
import logging
from PIL import Image
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding plots in Qt widgets."""

    def __init__(self, figure=None, width=5, height=4, dpi=100):
        """
        Initialize matplotlib canvas.

        Args:
            figure: Matplotlib figure (or creates a new one)
            width: Figure width in inches
            height: Figure height in inches
            dpi: DPI for rendering
        """
        if figure is None:
            figure = Figure(figsize=(width, height), dpi=dpi)

        self.figure = figure
        super().__init__(self.figure)


class ParticleVisualization:
    """Static visualization tools for particle data and tracking results."""

    @staticmethod
    def visualize_frame(frame: np.ndarray,
                       ax: Optional[plt.Axes] = None,
                       cmap: str = 'gray',
                       figsize: Tuple[int, int] = (10, 8),
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       colorbar: bool = True,
                       title: Optional[str] = None) -> plt.Figure:
        """
        Visualize a single microscopy frame.

        Args:
            frame: 2D frame data
            ax: Existing axes to plot on
            cmap: Colormap name
            figsize: Figure size (width, height) in inches
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            colorbar: Whether to add a colorbar
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot the frame with origin='lower'
        im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        # Add colorbar
        if colorbar:
            fig.colorbar(im, ax=ax, label='Intensity')

        # Add title
        if title:
            ax.set_title(title)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set aspect ratio
        ax.set_aspect('equal')

        return fig

    @staticmethod
    def overlay_positions(frame: np.ndarray,
                         positions: Union[np.ndarray, List[Tuple[float, float]]],
                         ax: Optional[plt.Axes] = None,
                         figsize: Tuple[int, int] = (10, 8),
                         marker: str = 'o',
                         color: str = 'red',
                         size: int = 20,
                         alpha: float = 0.7,
                         cmap: str = 'gray',
                         title: Optional[str] = None) -> plt.Figure:
        """
        Visualize a frame with overlaid particle positions.

        Args:
            frame: 2D frame data
            positions: Particle positions as array of (y,x) coordinates or 2D binary mask
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            marker: Marker type for positions
            color: Marker color
            size: Marker size
            alpha: Marker transparency
            cmap: Colormap for the frame
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot the frame with origin='lower'
        ax.imshow(frame, cmap=cmap, origin='lower')

        # Process positions
        if isinstance(positions, np.ndarray) and positions.ndim == 2 and positions.shape == frame.shape:
            # Binary mask
            y_coords, x_coords = np.where(positions > 0.5)
            x_pos = x_coords
            y_pos = y_coords
        else:
            # List of coordinates in (y, x) order
            if isinstance(positions, list):
                positions = np.array(positions)

            y_pos = positions[:, 0]
            x_pos = positions[:, 1]

        # Plot positions
        ax.scatter(x_pos, y_pos, marker=marker, c=color, s=size, alpha=alpha)

        # Add title
        if title:
            ax.set_title(title)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    @staticmethod
    def visualize_tracks(frames: np.ndarray,
                        tracks: List[np.ndarray],
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        frame_idx: int = 0,
                        cmap: str = 'gray',
                        line_alpha: float = 0.7,
                        point_size: int = 20,
                        title: Optional[str] = None) -> plt.Figure:
        """
        Visualize particle tracks on a frame.

        Args:
            frames: 3D array of frames (time, height, width)
            tracks: List of track arrays, each with shape (num_frames, 2) for (y,x) coordinates
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            frame_idx: Index of frame to display
            cmap: Colormap for the frame
            line_alpha: Track line transparency
            point_size: Current position point size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot the frame with origin='lower'
        ax.imshow(frames[frame_idx], cmap=cmap, origin='lower')

        # Generate colors for tracks
        colors = plt.cm.jet(np.linspace(0, 1, len(tracks)))

        # Plot each track
        for track_idx, track in enumerate(tracks):
            color = colors[track_idx]

            # Plot past positions
            past_mask = np.arange(len(track)) <= frame_idx
            if np.any(past_mask):
                past_track = track[past_mask]
                # Use x=track[:,1], y=track[:,0] since tracks store (y,x) coordinates
                ax.plot(past_track[:, 1], past_track[:, 0], '-', color=color, alpha=line_alpha)

                # Highlight current position
                if frame_idx < len(track):
                    ax.plot(track[frame_idx, 1], track[frame_idx, 0], 'o',
                            color=color, markersize=point_size//3)

        # Add title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Frame {frame_idx}")

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    @staticmethod
    def compare_predictions(frame: np.ndarray,
                           ground_truth: np.ndarray,
                           prediction: np.ndarray,
                           ax: Optional[plt.Axes] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'gray',
                           gt_color: str = 'green',
                           pred_color: str = 'red',
                           marker_size: int = 20,
                           alpha: float = 0.7,
                           title: Optional[str] = None) -> plt.Figure:
        """
        Compare ground truth and predicted particle positions.

        Args:
            frame: 2D frame data
            ground_truth: Ground truth positions (binary mask or (y,x) coordinate list)
            prediction: Predicted positions (binary mask or (y,x) coordinate list)
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            cmap: Colormap for the frame
            gt_color: Ground truth marker color
            pred_color: Prediction marker color
            marker_size: Marker size
            alpha: Marker transparency
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot the frame with origin='lower'
        ax.imshow(frame, cmap=cmap, origin='lower')

        # Process ground truth positions
        if isinstance(ground_truth, np.ndarray) and ground_truth.ndim == 2 and ground_truth.shape == frame.shape:
            # Binary mask
            gt_y, gt_x = np.where(ground_truth > 0.5)
        else:
            # List of coordinates in (y, x) order
            if isinstance(ground_truth, list):
                ground_truth = np.array(ground_truth)

            gt_y = ground_truth[:, 0]
            gt_x = ground_truth[:, 1]

        # Process predicted positions
        if isinstance(prediction, np.ndarray) and prediction.ndim == 2 and prediction.shape == frame.shape:
            # Binary mask
            pred_y, pred_x = np.where(prediction > 0.5)
        else:
            # List of coordinates in (y, x) order
            if isinstance(prediction, list):
                prediction = np.array(prediction)

            pred_y = prediction[:, 0]
            pred_x = prediction[:, 1]

        # Plot ground truth positions (slightly larger marker to appear behind predictions)
        ax.scatter(gt_x, gt_y, marker='o', c=gt_color, s=marker_size*1.5, alpha=alpha, label='Ground Truth')

        # Plot predicted positions
        ax.scatter(pred_x, pred_y, marker='x', c=pred_color, s=marker_size, alpha=alpha, label='Prediction')

        # Add legend
        ax.legend(loc='upper right')

        # Add title
        if title:
            ax.set_title(title)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    @staticmethod
    def visualize_track_matches(frames: np.ndarray,
                               gt_tracks: List[np.ndarray],
                               pred_tracks: List[np.ndarray],
                               matches: List[Tuple[int, int]],
                               ax: Optional[plt.Axes] = None,
                               figsize: Tuple[int, int] = (10, 8),
                               frame_idx: int = 0,
                               cmap: str = 'gray',
                               gt_color: str = 'green',
                               pred_color: str = 'red',
                               match_color: str = 'blue',
                               line_alpha: float = 0.7,
                               title: Optional[str] = None) -> plt.Figure:
        """
        Visualize matched tracks between ground truth and predictions.

        Args:
            frames: 3D array of frames (time, height, width)
            gt_tracks: List of ground truth track arrays in (y,x) order
            pred_tracks: List of predicted track arrays in (y,x) order
            matches: List of (gt_idx, pred_idx) tuples for matches
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            frame_idx: Index of frame to display
            cmap: Colormap for the frame
            gt_color: Ground truth track color
            pred_color: Prediction track color
            match_color: Matched track color
            line_alpha: Track line transparency
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot the frame with origin='lower'
        ax.imshow(frames[frame_idx], cmap=cmap, origin='lower')

        # Convert matches to sets for faster lookup
        gt_matched = set([m[0] for m in matches])
        pred_matched = set([m[1] for m in matches])

        # Plot unmatched ground truth tracks
        for idx, track in enumerate(gt_tracks):
            if idx not in gt_matched and frame_idx < len(track):
                # Plot past positions
                past_mask = np.arange(len(track)) <= frame_idx
                if np.any(past_mask):
                    past_track = track[past_mask]
                    # Use x=track[:,1], y=track[:,0] since tracks store (y,x) coordinates
                    ax.plot(past_track[:, 1], past_track[:, 0], '-',
                            color=gt_color, alpha=line_alpha, linewidth=1)

                    # Highlight current position
                    ax.plot(track[frame_idx, 1], track[frame_idx, 0], 'o',
                            color=gt_color, markersize=5)

        # Plot unmatched predicted tracks
        for idx, track in enumerate(pred_tracks):
            if idx not in pred_matched and frame_idx < len(track):
                # Plot past positions
                past_mask = np.arange(len(track)) <= frame_idx
                if np.any(past_mask):
                    past_track = track[past_mask]
                    # Use x=track[:,1], y=track[:,0] since tracks store (y,x) coordinates
                    ax.plot(past_track[:, 1], past_track[:, 0], '--',
                            color=pred_color, alpha=line_alpha, linewidth=1)

                    # Highlight current position
                    ax.plot(track[frame_idx, 1], track[frame_idx, 0], 'x',
                            color=pred_color, markersize=5)

        # Plot matched tracks
        for gt_idx, pred_idx in matches:
            gt_track = gt_tracks[gt_idx]
            pred_track = pred_tracks[pred_idx]

            if frame_idx < len(gt_track) and frame_idx < len(pred_track):
                # Plot past positions for ground truth
                past_mask = np.arange(len(gt_track)) <= frame_idx
                if np.any(past_mask):
                    past_track = gt_track[past_mask]
                    # Use x=track[:,1], y=track[:,0] since tracks store (y,x) coordinates
                    ax.plot(past_track[:, 1], past_track[:, 0], '-',
                            color=match_color, alpha=line_alpha, linewidth=1.5)

                # Plot current position for ground truth
                ax.plot(gt_track[frame_idx, 1], gt_track[frame_idx, 0], 'o',
                        color=match_color, markersize=7)

                # Plot current position for prediction
                ax.plot(pred_track[frame_idx, 1], pred_track[frame_idx, 0], 'x',
                        color=match_color, markersize=7)

                # Draw line connecting current positions
                ax.plot([gt_track[frame_idx, 1], pred_track[frame_idx, 1]],
                        [gt_track[frame_idx, 0], pred_track[frame_idx, 0]],
                        '--', color=match_color, alpha=0.5, linewidth=0.5)

        # Add legend
        gt_patch = mpatches.Patch(color=gt_color, label='Ground Truth')
        pred_patch = mpatches.Patch(color=pred_color, label='Prediction')
        match_patch = mpatches.Patch(color=match_color, label='Matched')
        ax.legend(handles=[gt_patch, pred_patch, match_patch], loc='upper right')

        # Add title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Frame {frame_idx}")

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    @staticmethod
    def visualize_probability_map(frame: np.ndarray,
                                 prob_map: np.ndarray,
                                 ax: Optional[plt.Axes] = None,
                                 figsize: Tuple[int, int] = (15, 10),
                                 cmap_frame: str = 'gray',
                                 cmap_prob: str = 'hot',
                                 blend_alpha: float = 0.5,
                                 colorbar: bool = True,
                                 title: Optional[str] = None) -> plt.Figure:
        """
        Visualize a probability map overlaid on a frame.

        Args:
            frame: 2D frame data
            prob_map: Probability map with same shape as frame
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            cmap_frame: Colormap for the frame
            cmap_prob: Colormap for the probability map
            blend_alpha: Blend transparency
            colorbar: Whether to add a colorbar
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Ensure shapes match
        if frame.shape != prob_map.shape:
            raise ValueError(f"Frame shape {frame.shape} doesn't match probability map shape {prob_map.shape}")

        # Plot the frame with origin='lower'
        ax.imshow(frame, cmap=cmap_frame, origin='lower')

        # Plot the probability map with origin='lower'
        im = ax.imshow(prob_map, cmap=cmap_prob, alpha=blend_alpha, origin='lower')

        # Add colorbar
        if colorbar:
            fig.colorbar(im, ax=ax, label='Probability')

        # Add title
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Probability Map")

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig


class TrackingAnimator:
    """Class for creating animations of particle tracks."""

    def __init__(self, frames: np.ndarray, tracks: List[np.ndarray]):
        """
        Initialize the tracking animator.

        Args:
            frames: 3D array of frames (time, height, width)
            tracks: List of track arrays, each with shape (num_frames, 2) for (y,x) coordinates
        """
        self.frames = frames
        self.tracks = tracks
        self.num_frames = len(frames)
        self.track_history = 10  # Number of frames to show in track history

        # Generate colors for tracks
        self.track_colors = plt.cm.jet(np.linspace(0, 1, len(tracks)))

    def create_animation(self,
                        figsize: Tuple[int, int] = (10, 8),
                        cmap: str = 'gray',
                        line_alpha: float = 0.7,
                        point_size: int = 20,
                        interval: int = 100,
                        history_length: int = 10,
                        save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Create an animation of particle tracks.

        Args:
            figsize: Figure size (width, height) in inches
            cmap: Colormap for the frames
            line_alpha: Track line transparency
            point_size: Current position point size
            interval: Animation interval in milliseconds
            history_length: Number of frames to show in track history
            save_path: Path to save the animation (None for no saving)

        Returns:
            Animation object
        """
        # Update history length
        self.track_history = history_length

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Plot initial frame with origin='lower'
        im = ax.imshow(self.frames[0], cmap=cmap, origin='lower')

        # Initialize line and point objects for each track
        lines = []
        points = []

        for track_idx, track in enumerate(self.tracks):
            color = self.track_colors[track_idx]
            line, = ax.plot([], [], '-', color=color, alpha=line_alpha)
            point, = ax.plot([], [], 'o', color=color, markersize=point_size//3)

            lines.append(line)
            points.append(point)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title
        title = ax.set_title("Frame 0")

        # Animation update function
        def update(frame_idx):
            # Update frame
            im.set_array(self.frames[frame_idx])

            # Update title
            title.set_text(f"Frame {frame_idx}")

            # Update track lines and points
            for track_idx, track in enumerate(self.tracks):
                # Check if track exists at current frame
                if frame_idx < len(track):
                    # Calculate history start index
                    start_idx = max(0, frame_idx - self.track_history + 1)

                    # Get track history
                    history = track[start_idx:frame_idx+1]

                    # Update line - use x=history[:,1], y=history[:,0] since tracks store (y,x) coordinates
                    lines[track_idx].set_data(history[:, 1], history[:, 0])

                    # Update current position
                    points[track_idx].set_data(track[frame_idx, 1], track[frame_idx, 0])
                else:
                    # Hide line and point if track doesn't exist
                    lines[track_idx].set_data([], [])
                    points[track_idx].set_data([], [])

            return [im, title] + lines + points

        # Create animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=range(self.num_frames),
            interval=interval,
            blit=True
        )

        # Save animation if requested
        if save_path is not None:
            anim.save(save_path)
            logger.info(f"Animation saved to {save_path}")

        return anim

    def save_animation_frames(self,
                             output_dir: str,
                             figsize: Tuple[int, int] = (10, 8),
                             cmap: str = 'gray',
                             line_alpha: float = 0.7,
                             point_size: int = 20,
                             history_length: int = 10,
                             format: str = 'png',
                             dpi: int = 100) -> List[str]:
        """
        Save individual frames of the animation.

        Args:
            output_dir: Directory to save frames
            figsize: Figure size (width, height) in inches
            cmap: Colormap for the frames
            line_alpha: Track line transparency
            point_size: Current position point size
            history_length: Number of frames to show in track history
            format: Image format ('png', 'jpg', etc.)
            dpi: DPI for saving images

        Returns:
            List of saved frame paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Update history length
        self.track_history = history_length

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # List to store frame paths
        frame_paths = []

        # Generate each frame
        for frame_idx in range(self.num_frames):
            # Clear axes
            ax.clear()

            # Plot frame with origin='lower'
            ax.imshow(self.frames[frame_idx], cmap=cmap, origin='lower')

            # Plot tracks
            for track_idx, track in enumerate(self.tracks):
                color = self.track_colors[track_idx]

                # Check if track exists at current frame
                if frame_idx < len(track):
                    # Calculate history start index
                    start_idx = max(0, frame_idx - self.track_history + 1)

                    # Get track history
                    history = track[start_idx:frame_idx+1]

                    # Plot track history - use x=history[:,1], y=history[:,0] since tracks store (y,x) coordinates
                    if len(history) > 1:
                        ax.plot(history[:, 1], history[:, 0], '-', color=color, alpha=line_alpha)

                    # Plot current position
                    ax.plot(track[frame_idx, 1], track[frame_idx, 0], 'o',
                            color=color, markersize=point_size//3)

            # Set title
            ax.set_title(f"Frame {frame_idx}")

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.{format}")
            fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            frame_paths.append(frame_path)

        # Close figure
        plt.close(fig)

        logger.info(f"Saved {len(frame_paths)} frames to {output_dir}")

        return frame_paths


class TrackingVisualizer(QWidget):
    """Interactive Qt widget for visualizing particle tracks."""

    frameChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialize the tracking visualizer widget."""
        super().__init__(parent)

        # Data
        self.frames = None
        self.tracks = None
        self.num_frames = 0
        self.track_history = 10
        self.track_colors = None
        self.current_frame = 0
        self.play_speed = 100  # ms
        self.is_playing = False

        # Set up UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Canvas for plotting
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = MatplotlibCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Control layout
        control_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_button)

        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        control_layout.addWidget(self.frame_slider)

        # Frame counter label
        self.frame_label = QLabel("Frame: 0/0")
        control_layout.addWidget(self.frame_label)

        # History length dropdown
        control_layout.addWidget(QLabel("History Length:"))
        self.history_combo = QComboBox()
        self.history_combo.addItems(['5', '10', '15', '20', 'All'])
        self.history_combo.setCurrentText(str(self.track_history))
        self.history_combo.currentTextChanged.connect(self.on_history_change)
        control_layout.addWidget(self.history_combo)

        # Speed dropdown
        control_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['Slow', 'Normal', 'Fast'])
        self.speed_combo.setCurrentText('Normal')
        self.speed_combo.currentTextChanged.connect(self.on_speed_change)
        control_layout.addWidget(self.speed_combo)

        # Add controls to main layout
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(control_layout)

        # Create timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.setLayout(main_layout)

    def set_data(self, frames: np.ndarray, tracks: List[np.ndarray]):
        """
        Set the data for visualization.

        Args:
            frames: 3D array of frames (time, height, width)
            tracks: List of track arrays, each with shape (num_frames, 2) for (y,x) coordinates
        """
        self.frames = frames
        self.tracks = tracks
        self.num_frames = len(frames)

        # Generate colors for tracks
        self.track_colors = plt.cm.jet(np.linspace(0, 1, len(tracks)))

        # Update slider
        self.frame_slider.setMaximum(self.num_frames - 1)

        # Reset to first frame
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"Frame: {self.current_frame}/{self.num_frames-1}")

        # Update frame
        self.update_frame()

    def update_frame(self):
        """Update the displayed frame."""
        if self.frames is None or self.current_frame >= self.num_frames:
            return

        # Clear the axes
        self.ax.clear()

        # Plot the frame with origin='lower'
        self.ax.imshow(self.frames[self.current_frame], cmap='gray', origin='lower')

        # Plot tracks
        for track_idx, track in enumerate(self.tracks):
            color = self.track_colors[track_idx]

            # Check if track exists at current frame
            if self.current_frame < len(track):
                # Calculate history start index
                if self.history_combo.currentText() == 'All':
                    start_idx = 0
                else:
                    history_length = int(self.history_combo.currentText())
                    start_idx = max(0, self.current_frame - history_length + 1)

                # Get track history
                history = track[start_idx:self.current_frame+1]

                # Plot track history - use x=history[:,1], y=history[:,0] since tracks store (y,x) coordinates
                if len(history) > 1:
                    self.ax.plot(history[:, 1], history[:, 0], '-', color=color, alpha=0.7)

                # Plot current position
                self.ax.plot(track[self.current_frame, 1], track[self.current_frame, 0], 'o',
                        color=color, markersize=7)

        # Set title
        self.ax.set_title(f"Frame {self.current_frame}")

        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Update canvas
        self.canvas.draw()

        # Emit signal
        self.frameChanged.emit(self.current_frame)

    def on_frame_change(self, value):
        """Handle frame slider value change."""
        self.current_frame = value
        self.frame_label.setText(f"Frame: {self.current_frame}/{self.num_frames-1}")
        self.update_frame()

    def on_history_change(self, value):
        """Handle history length change."""
        if value == 'All':
            self.track_history = self.num_frames
        else:
            self.track_history = int(value)
        self.update_frame()

    def on_speed_change(self, value):
        """Handle playback speed change."""
        if value == 'Slow':
            self.play_speed = 200
        elif value == 'Normal':
            self.play_speed = 100
        elif value == 'Fast':
            self.play_speed = 50

        if self.is_playing:
            self.timer.setInterval(self.play_speed)

    def toggle_play(self):
        """Toggle play/pause state."""
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(self.play_speed)
            self.play_button.setText("Pause")

        self.is_playing = not self.is_playing

    def next_frame(self):
        """Advance to the next frame."""
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
        else:
            # Loop back to the beginning
            self.current_frame = 0
            self.frame_slider.setValue(self.current_frame)

    def get_frame_as_image(self, format='png'):
        """
        Get the current frame as an image.

        Args:
            format: Image format (e.g., 'png', 'jpg')

        Returns:
            PIL Image object
        """
        buffer = io.BytesIO()
        self.figure.savefig(buffer, format=format, bbox_inches='tight')
        buffer.seek(0)
        return Image.open(buffer)


class TrainingProgressVisualization:
    """Visualization tools for model training progress."""

    @staticmethod
    def plot_losses(train_losses: List[float],
                   val_losses: Optional[List[float]] = None,
                   ax: Optional[plt.Axes] = None,
                   figsize: Tuple[int, int] = (10, 6),
                   title: str = "Training and Validation Loss") -> plt.Figure:
        """
        Plot training and validation losses.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot training loss
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')

        # Plot validation loss if provided
        if val_losses is not None:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        return fig

    @staticmethod
    def plot_learning_rate(learning_rates: List[float],
                          ax: Optional[plt.Axes] = None,
                          figsize: Tuple[int, int] = (10, 6),
                          title: str = "Learning Rate Schedule") -> plt.Figure:
        """
        Plot learning rate schedule.

        Args:
            learning_rates: List of learning rates
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot learning rate
        epochs = range(1, len(learning_rates) + 1)
        ax.plot(epochs, learning_rates, 'g-')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.grid(True)

        # Use log scale for y-axis
        ax.set_yscale('log')

        return fig

    @staticmethod
    def plot_metrics(metrics: Dict[str, List[float]],
                    ax: Optional[plt.Axes] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    title: str = "Training Metrics") -> plt.Figure:
        """
        Plot multiple training metrics.

        Args:
            metrics: Dictionary of metric names to lists of values
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot each metric
        for name, values in metrics.items():
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name)

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        return fig

    @staticmethod
    def compare_training_runs(runs: Dict[str, Dict[str, List[float]]],
                             metric: str = 'val_loss',
                             ax: Optional[plt.Axes] = None,
                             figsize: Tuple[int, int] = (10, 6),
                             title: Optional[str] = None) -> plt.Figure:
        """
        Compare the same metric across multiple training runs.

        Args:
            runs: Dictionary of run names to dictionaries of metrics
            metric: Name of the metric to compare
            ax: Existing axes to plot on
            figsize: Figure size (width, height) in inches
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot the metric for each run
        for run_name, metrics in runs.items():
            if metric in metrics:
                values = metrics[metric]
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, label=run_name)

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)

        if title is None:
            title = f"Comparison of {metric} across runs"

        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        return fig
