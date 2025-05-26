import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from torch import cuda, backends

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class SbertEncoder(BaseEncoder):
   
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: Optional[str] = None):
        
        super().__init__()
        if device is None:
            if backends.mps.is_available():
                device = "mps"
            elif cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        logger.info(f"Loading SBERT model '{model_name}' on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, sentences: List[str], lang: Optional[str] = None):
        logger.debug(f"Encoding {len(sentences)} sentences with SBERT model")
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings
