import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from .base_encoder import BaseEncoder


class SBERTEncoder(BaseEncoder):
    """
    Encoder using Sentence-BERT. Uses multilingual model by default.
    """
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_sentences(self, sentences: List[str], lang: str = None) -> np.ndarray:
        return self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
