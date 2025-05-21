import logging
from typing import List, Tuple

import faiss
import torch
import torch.nn.functional as F

from auto_align.encoders.encoder_factory import get_encoder

logger = logging.getLogger(__name__)


def align_sentences(source_sentences: List[str], target_sentences: List[str],
                   src_lang: str, tgt_lang: str, encoder_name: str = None,
                   threshold: float = 0.7, topk=5, batch_size=512) -> List[Tuple[int, int, float]]:
    """
    Align sentences from source and target lists using the specified encoder.
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

    src_emb = F.normalize(src_embeddings, dim=1).cpu().numpy().astype('float32')
    tgt_emb = F.normalize(tgt_embeddings, dim=1).cpu().numpy().astype('float32')

    d = src_emb.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(tgt_emb)

    aligned = []

    for i in range(0, len(src_emb), batch_size):
        block = src_emb[i:i + batch_size]
        D, I = idx.search(block, topk)
        for bi, row in enumerate(D):
            src_i = i + bi
            for rank, score in enumerate(row):
                if score < threshold:
                    break
                tgt_j = int(I[bi, rank])
                aligned.append((src_i, tgt_j, float(score)))

    return aligned
