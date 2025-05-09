import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from .base_encoder import BaseEncoder


class LaBSEEncoder(BaseEncoder):
    """
    Encoder using the LaBSE model from HuggingFace.
    """

    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # macOS M1/M2 optimized
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
        self.model = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(self.device)
        self.model.eval()

    def embed_sentences(self, sentences: List[str], lang: str) -> np.ndarray:
        """
        Embed a list of sentences into high-dimensional vector representations.

        Args:
            sentences (List[str]): List of sentences in the same language.
            lang (str): Language code (not used in LaBSE but required for compatibility).

        Returns:
            np.ndarray: Embedding matrix of shape (n_sentences, embedding_dim).
        """
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()
