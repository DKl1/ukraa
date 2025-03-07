"""
Module for alignment using FAISS.

This module defines the FAISSAligner class which wraps FAISS functions
to build an index and perform nearest-neighbor search for aligning sentence embeddings.
"""

import faiss
import numpy as np
from typing import Tuple


class FAISSAligner:
    def __init__(self, dimension: int):
        """
        Initialize the FAISSAligner with the given embedding dimension.

        Args:
            dimension (int): Dimensionality of the embeddings.
        """
        # Create a FAISS index using L2 (Euclidean) distance.
        self.index = faiss.IndexFlatL2(dimension)

    def build_index(self, embeddings: np.ndarray):
        """
        Build the FAISS index using the provided embeddings.

        Args:
            embeddings (np.ndarray): Array of shape (num_samples, dimension) with the embeddings.
        """
        self.index.add(embeddings)

    def align(self, query_embeddings: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align query embeddings by retrieving the top-k nearest neighbors.

        Args:
            query_embeddings (np.ndarray): Array of shape (num_queries, dimension).
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - distances: Array of shape (num_queries, k) with the L2 distances.
                - indices: Array of shape (num_queries, k) with the indices of the nearest neighbors.
        """
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
