import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

from auto_align.encoders.encoder_factory import get_encoder

logger = logging.getLogger(__name__)


def align_sentences_no_faiss(source_sentences: List[str], target_sentences: List[str],
                           src_lang: str, tgt_lang: str, encoder_name: str = None,
                           threshold: float = 0.7, topk=5, batch_size=512) -> List[Tuple[int, int, float]]:
    """
    Align sentences using direct cosine similarity computation without FAISS.
    Uses the same interface as the original align_sentences function.
    :param source_sentences: List of sentences in the source language.
    :param target_sentences: List of sentences in the target language.
    :param src_lang: Source language code (for encoders that require it).
    :param tgt_lang: Target language code.
    :param encoder_name: Optional encoder name to use (overrides default selection).
    :param threshold: Similarity threshold for considering a pair as aligned (0 <= threshold <= 1 for cosine similarity).
    :param topk: Number of nearest neighbors to consider.
    :param batch_size: Batch size for processing.
    :return: List of tuples (src_index, tgt_index, score) for each aligned pair,
             where indices refer to positions in the input lists, and score is the cosine similarity.
    """
    logger.info(f"Starting alignment: {len(source_sentences)} source sentences, {len(target_sentences)} target sentences")
    logger.info(f"Using encoder: {encoder_name or 'auto-selected'} (src_lang={src_lang}, tgt_lang={tgt_lang})")

    encoder = get_encoder(encoder_name, languages=(src_lang, tgt_lang))

    src_embeddings = encoder.encode(source_sentences, lang=src_lang)
    tgt_embeddings = encoder.encode(target_sentences, lang=tgt_lang)

    if not isinstance(src_embeddings, torch.Tensor):
        src_embeddings = torch.tensor(src_embeddings)
    if not isinstance(tgt_embeddings, torch.Tensor):
        tgt_embeddings = torch.tensor(tgt_embeddings)
    
    device = src_embeddings.device
    tgt_embeddings = tgt_embeddings.to(device)

    src_emb = F.normalize(src_embeddings, dim=1)
    tgt_emb = F.normalize(tgt_embeddings, dim=1)

    aligned = []

    for i in range(0, len(src_emb), batch_size):
        src_batch = src_emb[i:i + batch_size]
        
        similarity = torch.mm(src_batch, tgt_emb.t())
        
        for bi, scores in enumerate(similarity):
            src_i = i + bi
            
            topk_scores, topk_indices = torch.topk(scores, min(topk, len(scores)))
            
            for rank, (score, tgt_j) in enumerate(zip(topk_scores, topk_indices)):
                score = float(score)
                if score < threshold:
                    break
                aligned.append((src_i, int(tgt_j), score))

    return aligned 