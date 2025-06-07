
import argparse
import logging
import sys

from auto_align.data import parse_tmx
from auto_align import aligner, evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate aligner on TMX gold set")
    parser.add_argument("--tmx-file", "-x", required=True,
                        help="Path to TMX file with en↔uk segments")
    parser.add_argument("--src-lang", "-sl", required=True, help="Source lang code, e.g. en")
    parser.add_argument("--tgt-lang", "-tl", required=True, help="Target lang code, e.g. uk")
    parser.add_argument("--encoder", "-e", default=None,
                        help="Encoder name (labse, laser, laser2, sbert)")
    parser.add_argument("--threshold", "-th", type=float, default=0.7,
                        help="Cosine similarity threshold")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    src_sents, tgt_sents = parse_tmx(args.tmx_file, args.src_lang, args.tgt_lang)
    n = len(src_sents)
    if n != len(tgt_sents):
        logger.error("Кількість src != tgt у TMX!")
        sys.exit(1)

    gold_pairs = [(i, i) for i in range(n)]
    logger.info(f"Gold set: {n} пар (1–1 відповідність)")

    aligned = aligner.align_sentences(
        source_sentences=src_sents,
        target_sentences=tgt_sents,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        encoder_name=args.encoder,
        threshold=args.threshold
    )

    gold_pairs = set(gold_pairs)

    metrics = evaluation.evaluate_alignment(
        aligned,  
        src_sents,  
        tgt_sents, 
        gold_pairs 
    )

    print("\n=== Evaluation Results ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"TER:       {metrics['ter']:.4f}")
    print(f"BLEU:      {metrics['bleu']:.4f}")
    print(f"CHRF:      {metrics['chrf']:.4f}")
    print(f"BERT-P:    {metrics['bertscore_precision']:.4f}")
    print(f"BERT-R:    {metrics['bertscore_recall']:.4f}")
    print(f"BERT-F1:   {metrics['bertscore_f1']:.4f}")


if __name__ == "__main__":
    main()
