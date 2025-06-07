import logging
from typing import Optional, Tuple

from auto_align.constants.language_pairs_encoder import PREFERRED_ENCODER, DEFAULT_ENCODER
from auto_align.encoders.labse_encoder import LabseEncoder
from auto_align.encoders.sbert_encoder import SbertEncoder
from auto_align.encoders.laser_encoder_custom import LaserEncoder

logger = logging.getLogger(__name__)

_encoder_cache = {}


def get_encoder(encoder_name: Optional[str] = None, languages: Optional[Tuple[str, str]] = None):

    if encoder_name:
        key = encoder_name.strip().lower()
    else:
        key = None

    # Determine encoder key if not explicitly provided
    if key is None and languages:
        lang_pair = (languages[0].lower(), languages[1].lower())
        key = PREFERRED_ENCODER.get(lang_pair)

        if not key:
            key = PREFERRED_ENCODER.get((lang_pair[1], lang_pair[0]))
        if not key:
            key = DEFAULT_ENCODER
        logger.info(f"Automatically selected encoder '{key}' for language pair {languages}")
    elif key is None:
        key = DEFAULT_ENCODER
        logger.info(f"No encoder specified; using default '{key}'")

    key = key.lower()

    # Use caching to avoid duplicate model loads
    if key in _encoder_cache:
        logger.debug(f"Returning cached encoder instance for '{key}'")
        return _encoder_cache[key]

    encoder_instance = None
    if key == "labse":
        encoder_instance = LabseEncoder()
    elif key == "sbert":
        encoder_instance = SbertEncoder()
    elif key == "laser":
        encoder_instance = LaserEncoder()
    else:
        raise ValueError(f"Unknown encoder name '{encoder_name}'")

    _encoder_cache[key] = encoder_instance
    logger.info(f"Encoder '{key}' initialized and cached.")
    return encoder_instance
