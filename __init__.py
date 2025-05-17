"""Initialize the Deep Particle Tracker package."""

import os

# Create necessary directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

__version__ = '0.1.0'
