"""
Training pipeline for particle tracking models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from datetime import datetime

from PyQt5.QtCore import QObject, pyqtSignal

from models.network import create_model, ParticleTrackerModel
from models.losses import CombinedLoss
from utils.device_manager import device_manager
from utils.thread_manager import thread_manager

logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline for particle tracking models."""

    def __init__(self,
                 model_type: str = 'attentive',
                 model_config: Optional[Dict] = None,
                 optimizer_type: str = 'adam',
                 lr: float = 0.001,
                 weight_decay: float = 0.0001,
                 lambda_loc: float = 1.0,
                 lambda_track: float = 0.5,
                 scheduler_type: Optional[str] = 'plateau',
                 checkpoint_dir: str = 'checkpoints',
                 log_every: int = 10,
                 save_every: int = 100,
                 callbacks: Optional[List[Callable]] = None):
        """
        Initialize the trainer.

        Args:
            model_type: Type of model to train ('simple', 'dual', 'attentive')
            model_config: Model configuration dictionary
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            lr: Learning rate
            weight_decay: Weight decay for regularization
            lambda_loc: Weight for localization loss
            lambda_track: Weight for tracking loss
            scheduler_type: Type of learning rate scheduler (None, 'plateau', 'cosine', 'step')
            checkpoint_dir: Directory to save checkpoints
            log_every: Log every N steps
            save_every: Save checkpoint every N epochs
            callbacks: List of callback functions called after each epoch
        """
        self.model_type = model_type
        self.model_config = model_config or {}
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_loc = lambda_loc
        self.lambda_track = lambda_track
        self.scheduler_type = scheduler_type
        self.checkpoint_dir = checkpoint_dir
        self.log_every = log_every
        self.save_every = save_every
        self.callbacks = callbacks or []

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize model, optimizer, and loss
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.current_epoch = 0

        # Initialize devices
        self.device = device_manager.get_device()
        logger.info(f"Using device: {self.device}")

    def setup(self):
        """Set up model, optimizer, and loss function."""
        # Create model
        logger.info(f"Creating {self.model_type} model with config: {self.model_config}")
        self.model = create_model(self.model_type, **self.model_config)
        self.model = self.model.to(self.device)

        # Create optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Create criterion
        self.criterion = CombinedLoss(
            lambda_loc=self.lambda_loc,
            lambda_track=self.lambda_track
        )

        # Create scheduler
        if self.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10, eta_min=1e-6
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )

        logger.info("Model, optimizer, and loss function set up")

    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 50,
             resume: bool = False,
             log_dir: Optional[str] = None) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            resume: Whether to resume from checkpoint
            log_dir: Directory to save logs

        Returns:
            Dictionary with training history
        """
        # Set up the model if not done already
        if self.model is None:
            self.setup()

        # Create log directory if specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Resume from checkpoint if requested
        if resume:
            self._load_checkpoint('latest')

        # Training loop
        total_steps = len(train_loader) * epochs
        start_time = time.time()

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self._train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss = self._validate(val_loader)
            self.history['val_loss'].append(val_loss)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train loss: {train_loss:.6f}, "
                f"Val loss: {val_loss:.6f}, "
                f"Time: {time.time() - start_time:.2f}s"
            )

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or epoch == epochs - 1:
                self._save_checkpoint('latest')

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint('best')
                logger.info(f"New best validation loss: {val_loss:.6f}")

            # Call callbacks
            for callback in self.callbacks:
                callback(self, epoch, train_loss, val_loss)

        # Plot and save training history
        if log_dir:
            self._plot_training_history(log_dir)

        # Return training history
        return self.history

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
            for step, batch in enumerate(train_loader):
                # Move data to device
                frames = batch['frames'].to(self.device)

                # Debug logging for first few batches
                if step < 2 and epoch == 0:
                    logger.info(f"[Step {step}] Input frames shape: {frames.shape}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"[Step {step}] Batch {key} shape: {value.shape}")

                # Ensure frames have correct shape
                if frames.dim() == 5:  # (batch, frames, channels, height, width)
                    # Already in the right format for sequence models
                    pass
                elif frames.dim() == 4 and frames.size(1) <= 10:  # Likely (batch, frames, height, width)
                    frames = frames.unsqueeze(2)  # Add channels dimension
                elif frames.dim() == 4:  # Likely (batch, channels, height, width)
                    # Model is not sequence-based, maintain frame shape
                    pass  # Don't add frame dimension here
                elif frames.dim() == 3:  # (batch, height, width)
                    frames = frames.unsqueeze(1)  # Add channels dimension

                # Ensure float32 dtype
                if frames.dtype != torch.float32:
                    frames = frames.float()

                # Prepare targets
                targets = {}
                for key in batch.keys():
                    if key in ['positions', 'track_ids', 'masks']:
                        targets[key] = batch[key].to(self.device)

                # Forward pass with better error handling
                self.optimizer.zero_grad()
                try:
                    outputs = self.model(frames)

                    # Calculate loss with error handling
                    try:
                        losses = self.criterion(outputs, targets)
                        loss = losses['total']
                    except Exception as e:
                        logger.error(f"Error calculating loss: {str(e)}")
                        logger.error(f"Output type: {type(outputs)}")
                        if isinstance(outputs, dict):
                            for k, v in outputs.items():
                                logger.error(f"Output {k} shape: {v.shape}")
                        elif isinstance(outputs, torch.Tensor):
                            logger.error(f"Output tensor shape: {outputs.shape}")

                        for k, v in targets.items():
                            logger.error(f"Target {k} shape: {v.shape}")
                        raise e

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Update metrics
                    total_loss += loss.item()

                    # Update progress bar
                    if step % self.log_every == 0:
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'loc_loss': losses.get('loc_loss', torch.tensor(0.0)).item(),
                            'track_loss': losses.get('track_loss', torch.tensor(0.0)).item()
                        })
                except Exception as e:
                    logger.error(f"Error during forward/backward pass: {str(e)}")
                    if step == 0:
                        # Print detailed error info for debugging
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        logger.error(f"Input frames shape: {frames.shape}")
                    raise e

                pbar.update(1)

        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                frames = batch['frames'].to(self.device)

                # Ensure frames have shape (batch, frames, channels, height, width)
                # for sequence models or (batch, channels, height, width) for single frame
                if frames.dim() == 5:  # (batch, frames, channels, height, width)
                    # Already in the right format for sequence models
                    pass
                elif frames.dim() == 4 and self.model_type != 'simple':
                    # Probably (batch, frames, height, width) missing channels
                    frames = frames.unsqueeze(2)
                elif frames.dim() == 4:
                    # Probably (batch, channels, height, width) for single frame models
                    pass
                elif frames.dim() == 3:
                    # Probably (batch, height, width) missing channels
                    frames = frames.unsqueeze(1)

                # Ensure float32 dtype
                if frames.dtype != torch.float32:
                    frames = frames.float()

                # Prepare targets
                targets = {}
                for key in ['positions', 'track_ids']:
                    if key in batch:
                        target_tensor = batch[key].to(self.device)
                        # Ensure float32 for floating point tensors
                        if torch.is_floating_point(target_tensor):
                            target_tensor = target_tensor.float()
                        targets[key] = target_tensor

                # Forward pass
                outputs = self.model(frames)

                # Calculate loss
                losses = self.criterion(outputs, targets)
                loss = losses['total']

                # Update metrics
                total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(val_loader)

        return avg_loss

    def _save_checkpoint(self, name: str):
        """
        Save a checkpoint.

        Args:
            name: Checkpoint name
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'model_type': self.model_type,
            'model_config': self.model_config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        if name == 'latest':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        elif name == 'best':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{name}_{timestamp}.pth')

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, name: str) -> int:
        """
        Load a checkpoint.

        Args:
            name: Checkpoint name or path

        Returns:
            Current epoch
        """
        # Resolve checkpoint path
        if os.path.isfile(name):
            checkpoint_path = name
        elif name == 'latest':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        elif name == 'best':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{name}.pth')

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return 0

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Check if model type matches
        if 'model_type' in checkpoint and checkpoint['model_type'] != self.model_type:
            logger.warning(
                f"Model type mismatch: checkpoint has {checkpoint['model_type']}, "
                f"but current model is {self.model_type}"
            )

        # Check if model config matches
        if 'model_config' in checkpoint:
            # Only check keys that exist in both configs
            for key in set(checkpoint['model_config'].keys()).intersection(self.model_config.keys()):
                if checkpoint['model_config'][key] != self.model_config[key]:
                    logger.warning(
                        f"Model config mismatch for key {key}: "
                        f"checkpoint has {checkpoint['model_config'][key]}, "
                        f"but current config has {self.model_config[key]}"
                    )

        # Load model state if model exists
        if self.model is None:
            # Create model with checkpoint config
            model_config = checkpoint.get('model_config', {})
            self.model = create_model(checkpoint.get('model_type', self.model_type), **model_config)
            self.model = self.model.to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Create optimizer if not exists
        if self.optimizer is None:
            self.setup()

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load other attributes
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.current_epoch}")

        return self.current_epoch

    def _plot_training_history(self, log_dir: str):
        """
        Plot and save training history.

        Args:
            log_dir: Directory to save plots
        """
        # Set a non-interactive backend before creating any figures
        # This makes matplotlib thread-safe for saving figures
        import matplotlib
        orig_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Use the non-interactive Agg backend

        try:
            plt.figure(figsize=(12, 5))

            # Plot training and validation loss
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_loss'], label='Train')
            plt.plot(self.history['val_loss'], label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            # Plot learning rate if available
            if self.scheduler is not None:
                plt.subplot(1, 2, 2)
                lrs = []
                for param_group in self.optimizer.param_groups:
                    lrs.append(param_group['lr'])
                plt.plot(lrs)
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate')
                plt.grid(True)

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(log_dir, f'training_history_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close('all')  # Close all figures to free memory

            logger.info(f"Training history plot saved to {plot_path}")
        finally:
            # Restore the original backend
            matplotlib.use(orig_backend)


    def load_weights_from_checkpoint(self, checkpoint_path: str, strict: bool = False) -> bool:
        """
        Load only model weights from a checkpoint without loading optimizer state.

        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in state_dict match the model's keys

        Returns:
            Boolean indicating if loading was successful
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return False

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Check model compatibility
            if 'model_type' in checkpoint and checkpoint['model_type'] != self.model_type:
                logger.warning(
                    f"Model type mismatch: checkpoint has {checkpoint['model_type']}, "
                    f"but current model is {self.model_type}"
                )
                if strict:
                    return False

            # Check model config - verify key dimensions match
            if 'model_config' in checkpoint:
                for key in ['input_channels', 'output_channels', 'depth']:
                    if (key in checkpoint['model_config'] and key in self.model_config and
                        checkpoint['model_config'][key] != self.model_config[key]):
                        logger.warning(
                            f"Model config mismatch for key {key}: "
                            f"checkpoint has {checkpoint['model_config'][key]}, "
                            f"but current config has {self.model_config[key]}"
                        )
                        if strict:
                            return False

            # Load model state
            if 'model_state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
                    logger.info(f"Model weights loaded from {checkpoint_path}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading model weights: {str(e)}")
                    if strict:
                        return False

                    # Try loading with a partial match (useful for transfer learning)
                    if not strict:
                        logger.info("Attempting to load weights with partial match...")
                        # Get state dict from checkpoint and model
                        state_dict = checkpoint['model_state_dict']
                        model_dict = self.model.state_dict()

                        # Filter out incompatible keys
                        filtered_dict = {k: v for k, v in state_dict.items()
                                         if k in model_dict and v.shape == model_dict[k].shape}

                        # Load filtered state dict
                        model_dict.update(filtered_dict)
                        self.model.load_state_dict(model_dict)

                        logger.info(f"Loaded {len(filtered_dict)}/{len(state_dict)} layers from checkpoint")
                        return len(filtered_dict) > 0
            else:
                logger.warning(f"No model state dictionary found in {checkpoint_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False

    def _temporal_embedding_loss(self, outputs, targets):
        """
        Calculate temporal consistency loss for particle embeddings.
        Encourages same particle to have similar embeddings across frames.
        """
        if 'track_ids' not in targets:
            return torch.tensor(0.0, device=self.device)

        track_ids = targets['track_ids']  # (batch, frames, 1, H, W)
        embeddings = outputs['embeddings']  # (batch, C, H, W)

        # Simple implementation: just ensure consistency between consecutive frames
        # In a complete implementation, this would be more sophisticated
        loss = torch.tensor(0.0, device=self.device)

        return loss


class TrainingManager:
    """Manager for training jobs with thread management."""

    # Add a signal that will be emitted when training completes
    training_completed = pyqtSignal(str, dict)  # training_id, results

    def __init__(self, checkpoint_dir: str = 'checkpoints', log_dir: str = 'logs'):
        """
        Initialize the training manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        super().__init__()  # Initialize QObject
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Ensure directories exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Current training state
        self.trainers = {}
        self.current_training_id = None
        self.training_results = {}

    def start_training(self,
                      training_id: str,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      model_type: str = 'attentive',
                      model_config: Optional[Dict] = None,
                      optimizer_type: str = 'adam',
                      lr: float = 0.001,
                      epochs: int = 50,
                      resume: bool = False,
                      callback: Optional[Callable] = None) -> str:
        """
        Start a training job in a background thread.

        Args:
            training_id: Unique ID for the training job
            train_loader: Training data loader
            val_loader: Validation data loader
            model_type: Type of model to train
            model_config: Model configuration
            optimizer_type: Type of optimizer
            lr: Learning rate
            epochs: Number of epochs
            resume: Whether to resume from checkpoint
            callback: Callback function to call after training

        Returns:
            Task ID of the training job
        """
        # Create trainer
        trainer = self.create_trainer(
            training_id=training_id,
            model_type=model_type,
            model_config=model_config,
            optimizer_type=optimizer_type,
            lr=lr,
            scheduler_type='plateau'
        )

        # Start training with the trainer
        return self.start_training_with_trainer(
            trainer=trainer,
            training_id=training_id,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            resume=resume,
            callback=callback
        )

    def stop_training(self, training_id: str) -> bool:
        """
        Stop a training job.

        Args:
            training_id: ID of the training job to stop

        Returns:
            Whether the job was successfully stopped
        """
        task_id = f"train_{training_id}"

        # Try to cancel the task
        cancelled = thread_manager.cancel_task(task_id)

        if cancelled:
            logger.info(f"Training job {training_id} stopped")
        else:
            logger.warning(f"Failed to stop training job {training_id}")

        return cancelled

    def get_training_status(self, training_id: str) -> Dict:
        """
        Get the status of a training job.

        Args:
            training_id: ID of the training job

        Returns:
            Status dictionary
        """
        task_id = f"train_{training_id}"
        task_status = thread_manager.get_task_status(task_id)

        # Get trainer if available
        trainer = self.trainers.get(training_id)

        status = {
            'task_status': task_status['status'],
            'training_id': training_id
        }

        if trainer:
            status.update({
                'current_epoch': trainer.current_epoch,
                'best_val_loss': trainer.best_val_loss,
                'history': trainer.history
            })

        return status

    def list_trainings(self) -> List[Dict]:
        """
        List all training jobs.

        Returns:
            List of training job statuses
        """
        return [self.get_training_status(training_id) for training_id in self.trainers.keys()]

    def load_model(self,
                 training_id: str,
                 checkpoint: str = 'best',
                 model_type: Optional[str] = None,
                 model_config: Optional[Dict] = None) -> nn.Module:
        """
        Load a trained model.

        Args:
            training_id: ID of the training job
            checkpoint: Checkpoint name ('best', 'latest', or specific path)
            model_type: Model type (if not loading from checkpoint)
            model_config: Model configuration (if not loading from checkpoint)

        Returns:
            Loaded model
        """
        # Resolve checkpoint path
        if os.path.isfile(checkpoint):
            checkpoint_path = checkpoint
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, training_id, f'{checkpoint}.pth')

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint {checkpoint_path} not found")

        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device_manager.get_device())

        # Create model
        if model_type is None:
            model_type = checkpoint_data.get('model_type', 'attentive')

        if model_config is None:
            model_config = checkpoint_data.get('model_config', {})

        model = create_model(model_type, **model_config)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model = model.to(device_manager.get_device())
        model.eval()

        logger.info(f"Model loaded from checkpoint {checkpoint_path}")

        return model

    def cleanup(self):
        """Clean up resources used by the training manager."""
        for training_id in list(self.trainers.keys()):
            self.stop_training(training_id)

        self.trainers.clear()
        self.training_results.clear()
        self.current_training_id = None

    def start_training_with_trainer(self,
                                  trainer: Trainer,
                                  training_id: str,
                                  train_loader: DataLoader,
                                  val_loader: DataLoader,
                                  epochs: int = 50,
                                  resume: bool = False,
                                  callback: Optional[Callable] = None) -> str:
        """
        Start a training job in a background thread using an existing trainer.

        Args:
            trainer: Existing trainer instance
            training_id: Unique ID for the training job
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            resume: Whether to resume from checkpoint
            callback: Callback function to call after training

        Returns:
            Task ID of the training job
        """
        # Store callback for later use
        if callback:
            self.callbacks[training_id] = callback

        # Store trainer
        self.trainers[training_id] = trainer
        self.current_training_id = training_id


        # Define training function
        def train_job():
            try:
                logger.info(f"Starting training job {training_id}")

                # Train the model
                result = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    resume=resume,
                    log_dir=os.path.join(self.log_dir, training_id)
                )

                # Store result
                self.training_results[training_id] = result

                logger.info(f"Training job {training_id} completed")

                # Emit signal instead of directly calling callback
                self.training_completed.emit(training_id, result)

                return result

            except Exception as e:
                logger.error(f"Error in training job {training_id}: {str(e)}")
                raise e

        # Start training in a background thread
        task_id = thread_manager.submit_task(
            task_id=f"train_{training_id}",
            func=train_job
        )

        logger.info(f"Training job {training_id} started with task ID {task_id}")

        return task_id

    def create_trainer(self,
                     training_id: str,
                     model_type: str,
                     model_config: Dict,
                     optimizer_type: str = 'adam',
                     lr: float = 0.001,
                     scheduler_type: Optional[str] = 'plateau') -> Trainer:
        """
        Create a trainer instance without starting training.

        Args:
            training_id: Unique ID for the training job
            model_type: Type of model to train
            model_config: Model configuration
            optimizer_type: Type of optimizer
            lr: Learning rate
            scheduler_type: Type of learning rate scheduler

        Returns:
            Trainer instance
        """
        # Create checkpoint and log directories for this training job
        job_checkpoint_dir = os.path.join(self.checkpoint_dir, training_id)
        job_log_dir = os.path.join(self.log_dir, training_id)

        os.makedirs(job_checkpoint_dir, exist_ok=True)
        os.makedirs(job_log_dir, exist_ok=True)

        # Create trainer
        trainer = Trainer(
            model_type=model_type,
            model_config=model_config,
            optimizer_type=optimizer_type,
            lr=lr,
            scheduler_type=scheduler_type,
            checkpoint_dir=job_checkpoint_dir,
            log_every=10,
            save_every=1
        )

        # Initialize model and optimizer
        trainer.setup()

        return trainer
