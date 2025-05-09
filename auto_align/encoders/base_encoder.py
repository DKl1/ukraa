import numpy as np
from typing import List
from abc import ABC, abstractmethod


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
