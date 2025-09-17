"""
Centralized configuration for the Loguru logger.
"""

import sys

from loguru import logger


def setup_logging() -> None:
    """
    Configures the global loguru logger for the application.

    This function removes the default handler and adds a new one with a
    more informative format that includes timestamps, levels, and file locations.
    """
    logger.remove()  # Remove the default logger to prevent duplicate outputs
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.info("Logger configured successfully.")
