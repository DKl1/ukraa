import logging


def setup_logger(name: str = "ukraa") -> logging.Logger:

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
