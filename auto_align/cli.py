"""
Command-Line Interface (CLI) for the Automatic Text Alignment Tool.

This module integrates data loading, embedding generation, alignment,
and evaluation into a single end-to-end pipeline that can be run from the command line.
It supports [ ] language pair via parameters --src_lang and --tgt_lang.
"""

import argparse
import numpy as np
from auto_align.data import load_text
from auto_align.embeddings import LaserEncoder
from auto_align.aligner import FAISSAligner
from auto_align.evaluation import compute_metrics
from auto_align.utils import setup_logger


def main() -> None:
    """
    Main function to run the text alignment pipeline.

    Parses command-line arguments, loads data, computes embeddings,
    aligns sentences, saves output, and evaluates alignments if a gold file is provided.
    """
    parser = argparse.ArgumentParser(description="Automatic Text Alignment Tool")
    parser.add_argument("--src_file", type=str, required=True, help="Path to source language text file")
    parser.add_argument("--tgt_file", type=str, required=True, help="Path to target language text file")
    parser.add_argument("--gold_file", type=str, default=None, help="Path to gold standard indices file")
    parser.add_argument("--format", type=str, default="txt", help="File format (txt, csv)")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--src_lang", type=str, default="uk", help="Source language code (default: 'uk')")
    parser.add_argument("--tgt_lang", type=str, default="en", help="Target language code (default: 'en')")
    parser.add_argument("--metrics", nargs="+", default=["precision@1", "precision@k", "mrr"],
                        help="Evaluation metrics to compute")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Loading sentences...")
    src_sentences = load_text(args.src_file, fmt=args.format)
    tgt_sentences = load_text(args.tgt_file, fmt=args.format)

    encoder = LaserEncoder()
    logger.info("Embedding sentences...")
    src_embeddings = encoder.embed_sentences(src_sentences, lang=args.src_lang)
    tgt_embeddings = encoder.embed_sentences(tgt_sentences, lang=args.tgt_lang)

    dimension = src_embeddings.shape[1]
    aligner = FAISSAligner(dimension)
    aligner.build_index(src_embeddings)

    logger.info("Aligning sentences...")
    distances, indices = aligner.align(tgt_embeddings, k=args.k)

    # Save aligned pairs
    output_filename = "aligned_output.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        for i in range(len(tgt_sentences)):
            best_idx = indices[i, 0]
            f.write(f"Source ({args.src_lang}): {src_sentences[best_idx]} || "
                    f"Target ({args.tgt_lang}): {tgt_sentences[i]} || "
                    f"Distance: {distances[i, 0]:.4f}\n")
    logger.info(f"Aligned output saved to {output_filename}")

    # Evaluate if gold standard is provided
    if args.gold_file:
        gold = np.loadtxt(args.gold_file, dtype=int)
        metrics_results = compute_metrics(indices, gold, metrics=args.metrics, k=args.k)
        for metric, value in metrics_results.items():
            logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
