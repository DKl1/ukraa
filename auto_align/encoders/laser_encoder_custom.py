import numpy as np
from typing import List
from laserembeddings import Laser
from .base_encoder import BaseEncoder


class LaserEncoder(BaseEncoder):
    def __init__(self) -> None:
        self.laser = Laser()

    def embed_sentences(self, sentences: List[str], lang: str) -> np.ndarray:
        """
        Generate embeddings for the given sentences using LASER.

        Args:
            sentences (List[str]): List of sentences.
            lang (str): Language code.

        Returns:
            np.ndarray: Array of embeddings.
        """
        return self.laser.embed_sentences(sentences, lang=lang)
