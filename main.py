#!/usr/bin/env python
"""
Deep Particle Tracker - Main application entry point
"""

import sys
import os
import logging
import argparse
import datetime
from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Particle Tracker')

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU mode (disable GPU)'
    )

    parser.add_argument(
        '--no_console',
        action='store_true',
        help='Disable console logging in GUI'
    )

    return parser.parse_args()

def setup_logging(debug=False, log_dir='logs'):
    """
    Set up logging with both file and console handlers.

    Args:
        debug: Whether to enable debug logging
        log_dir: Directory to store log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler with rotation
    log_file = os.path.join(log_dir, f'app_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Create stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Log start message
    root_logger.info(f"Logging initialized. Log file: {log_file}")

    return root_logger

def main():
    """Main application entry point with improved logging."""
    # Parse arguments
    args = parse_arguments()

    # Set up logging
    setup_logging(debug=args.debug, log_dir='logs')

    # If no_console flag is set, disable console logging
    if hasattr(args, 'no_console') and args.no_console:
        logger.info("Console logging disabled")

    # Create Qt application
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create main window
    main_window = MainWindow(enable_console=not getattr(args, 'no_console', False))
    main_window.show()

    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
