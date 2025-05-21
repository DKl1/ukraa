import logging
import torch
from typing import List, Optional

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class LaserEncoder(BaseEncoder):
    def __init__(self):
  
        super().__init__()
        try:
            from laserembeddings import Laser
        except ImportError as e:
            raise ImportError("LaserEncoder requires the 'laserembeddings' package. "
                              "Install it with `pip install laserembeddings`.") from e

        self._laser = Laser()
        logger.info("LASER model loaded successfully.")

    def encode(self, sentences: List[str], lang: Optional[str] = None):

        if lang is None:
            raise ValueError("Language code must be specified when using LASER encoder.")

        embeddings = self._laser.embed_sentences(sentences, lang=lang)

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        embeddings_tensor = torch.tensor(embeddings)
        embeddings_tensor = embeddings_tensor.to(device)

        return embeddings_tensor
