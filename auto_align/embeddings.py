"""
Module for generating sentence embeddings using different models.

This module defines an abstract base class for encoders and implements a LASER-based encoder.
"""

import numpy as np
from typing import List
from abc import ABC, abstractmethod
from laserembeddings import Laser


class BaseEncoder(ABC):

    @abstractmethod
    def embed_sentences(self, sentences: List[str], lang: str) -> np.ndarray:
        """
        Abstract method to generate embeddings for a list of sentences for a given language.

        Args:
            sentences (List[str]): List of sentences.
            lang (str): Language code.

        Returns:
            np.ndarray: Array of embeddings.
        """
        pass


class LaserEncoder(BaseEncoder):
    def __init__(self) -> None:
        """Initialize the LaserEncoder by loading the LASER model."""
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
