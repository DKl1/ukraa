import logging
from typing import List, Tuple, Set, Dict

import sacrebleu
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


def load_gold_alignment(gold_file_path: str) -> Set[Tuple[str, str]]:

    gold_pairs = set()
    with open(gold_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                src, tgt = line.split('\t', 1)
            elif '|||' in line:
                src, tgt = line.split('|||', 1)
                src = src.strip()
                tgt = tgt.strip()
            else:
                logger.warning(f"Gold alignment line not recognized format: {line}")
                continue
            gold_pairs.add((src, tgt))
    logger.info(f"Loaded {len(gold_pairs)} gold aligned pairs from {gold_file_path}")
    return gold_pairs


def evaluate_alignment(predicted_pairs: List[Tuple[int, int, float]],
                       source_sentences: List[str],
                       target_sentences: List[str],
                       gold_pairs: Set[Tuple[str, str]]) -> Dict[str, float]:
   
    gold_pairs = set(gold_pairs)

    pred_pairs_text = {(source_sentences[i].strip(), target_sentences[j].strip()) for i, j, _ in predicted_pairs}
    logger.info(f"Predicted pairs count = {len(predicted_pairs)}")

    if not gold_pairs:
        logger.warning("No gold pairs provided. Cannot compute metrics.")
        return {}

    true_positives = pred_pairs_text & gold_pairs
    num_pred = len(pred_pairs_text)
    num_gold = len(gold_pairs)
    num_tp = len(true_positives)

    precision = num_tp / num_pred if num_pred > 0 else 0.0
    recall = num_tp / num_gold if num_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


    ref_texts = []
    hyp_texts = []

    for src, tgt in gold_pairs:
        for pred_src, pred_tgt in pred_pairs_text:
            if src == pred_src:
                ref_texts.append(tgt)
                hyp_texts.append(pred_tgt)
                break

    if not hyp_texts:
        logger.warning("No overlap between gold and predicted sources for text-based metrics.")
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ter': 1.0,
            'bleu': 0.0,
            'chrf': 0.0,
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
        }

    ter = sacrebleu.metrics.TER().corpus_score(hyp_texts, [ref_texts]).score / 100
    bleu = sacrebleu.corpus_bleu(hyp_texts, [ref_texts]).score / 100
    chrf = sacrebleu.metrics.CHRF().corpus_score(hyp_texts, [ref_texts]).score / 100

    P, R, F1 = bert_score(hyp_texts, ref_texts, lang="en", verbose=False)
    bert_p = P.mean().item()
    bert_r = R.mean().item()
    bert_f1 = F1.mean().item()

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ter': ter,
        'bleu': bleu,
        'chrf': chrf,
        'bertscore_precision': bert_p,
        'bertscore_recall': bert_r,
        'bertscore_f1': bert_f1
    }

    for key, value in metrics.items():
        logger.info(f"{key.upper()}: {value:.3f}")

    return metrics
