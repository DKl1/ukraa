"""
Utility module for common functions.

This module includes a function for setting up a logger with a consistent format.
"""
import logging


def setup_logger(name: str = "ukraa") -> logging.Logger:
    """
    Set up a logger with a given name and a standardized format.

    Args:
        name (str, optional): The name of the logger. Defaults to "ukraa".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
