import argparse
import logging
import sys
from pathlib import Path

from auto_align import aligner, evaluation
from auto_align.data import parse_tmx, load_and_preprocess

import nltk
nltk.download('punkt_tab')

def main():
    parser = argparse.ArgumentParser(prog="ukraa-align", 
                                     description="Align sentences from a source and target text file using UKRAA aligner.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--src-file", "-s", help="Plain-text source file")
    group.add_argument("--tmx-file", "-x", help="TMX file (contains both sides)")
    parser.add_argument("--tgt-file", "-t", help="Plain-text target file (ignored if --tmx-file is used)")
    parser.add_argument("--src-lang", "-sl", help="Source language code (e.g. 'en'). Required for LASER/LASER2 encoders.")
    parser.add_argument("--tgt-lang", "-tl", help="Target language code (e.g. 'fr'). Required for LASER/LASER2 encoders.")
    parser.add_argument("--encoder", "-e", choices=["labse", "laser", "laser2", "sbert"], 
                        help="Which encoder to use. If not provided, selects automatically based on languages.")
    parser.add_argument("--threshold", "-th", type=float, default=0.7, 
                        help="Cosine similarity threshold for alignment (0 to 1). Default=0.7")
    parser.add_argument("--topk", "-k", type=int, default=5,
                        help="Number of nearest neighbors to consider for each source sentence. Default=5")
    parser.add_argument("--batch-size", "-b", type=int, default=512,
                        help="Batch size for processing embeddings. Default=512")
    parser.add_argument("--output", "-o", default="aligned_output.txt", help="Output file path for aligned pairs. Default='aligned_output.txt'")
    parser.add_argument("--gold", "-g", help="Path to gold alignment file (for evaluation). Optional.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (debug mode).")
    args = parser.parse_args()

    # Setup logging format and level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("UKRAA Sentence Aligner CLI started.")

    def infer_lang(path: Path):
        stem = path.stem.lower()
        return "".join(ch for ch in stem if ch.isalpha())

    # Determine language codes
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    if args.tmx_file:
        if not (args.src_lang and args.tgt_lang):
            logger.error("When using --tmx-file you must also specify --src-lang and --tgt-lang")
            sys.exit(1)
    else:
        if not args.src_file or not args.tgt_file:
            logger.error("Must supply both --src-file and --tgt-file (or use --tmx-file).")
            sys.exit(1)
        src_path = Path(args.src_file)
        tgt_path = Path(args.tgt_file)
        if not src_path.exists():
            logger.error(f"Source not found: {src_path}")
            sys.exit(1)
        if not tgt_path.exists():
            logger.error(f"Target not found: {tgt_path}")
            sys.exit(1)

        if not src_lang:
            src_lang = infer_lang(src_path)
        if not tgt_lang:
            tgt_lang = infer_lang(tgt_path)

    if not (src_lang and tgt_lang):
        logger.error("Could not infer languages; please supply --src-lang and --tgt-lang.")
        sys.exit(1)

    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()
    logger.debug(f"Using languages: src={src_lang}, tgt={tgt_lang}")

    if args.tmx_file:
        logger.info(f"Reading TMX file {args.tmx_file}")

    if args.tmx_file:
        logger.info(f"Parsing TMX: {args.tmx_file}")
        logger.info(f"Parsing TMX: {args.tmx_file}")
        source_sentences, target_sentences = parse_tmx(args.tmx_file, src_lang, tgt_lang)
    else:
        logger.info("Loading and preprocessing source…")
        source_sentences = load_and_preprocess(args.src_file)
        logger.info("Loading and preprocessing target…")
        target_sentences = load_and_preprocess(args.tgt_file)
    logger.info(f"Source: {len(source_sentences)} sentences after cleanup")
    logger.info(f"Target: {len(target_sentences)} sentences after cleanup")

    try:
        aligned_pairs = aligner.align_sentences(source_sentences, target_sentences,
                                               src_lang, tgt_lang,
                                               encoder_name=args.encoder,
                                               threshold=args.threshold,
                                               topk=args.topk,
                                               batch_size=args.batch_size)
    except ImportError as ie:
        logger.error(f"Alignment failed due to missing dependency or model: {ie}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during alignment: {e}")
        exit(1)

    output_path = Path(args.output)
    try:
        with open(output_path, 'w', encoding='utf-8') as out:
            for i, j, score in aligned_pairs:
                src_text = source_sentences[i]
                tgt_text = target_sentences[j]
                out.write(f"{src_text}\t{tgt_text}\t{score:.4f}\n")
        logger.info(f"Aligned pairs saved to {output_path} (total {len(aligned_pairs)} pairs).")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")

    if args.gold:
        gold_path = Path(args.gold)
        if not gold_path.exists():
            logger.error(f"Gold file not found: {gold_path}")
        else:
            gold_pairs = evaluation.load_gold_alignment(str(gold_path))
            metrics = evaluation.evaluate_alignment(
                aligned_pairs,
                source_sentences,
                target_sentences,
                gold_pairs
            )

            # extract the numbers
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            ter = metrics['ter']
            bleu = metrics['bleu']
            chrf = metrics['chrf']
            bert_p = metrics['bertscore_precision']
            bert_r = metrics['bertscore_recall']
            bert_f1 = metrics['bertscore_f1']

            with open(output_path, 'a', encoding='utf-8') as out:
                out.write(
                    f"# "
                    f"Precision={precision:.3f}, "
                    f"Recall={recall:.3f}, "
                    f"F1={f1:.3f}, "
                    f"TER={ter:.3f}, "
                    f"BLEU={bleu:.3f}, "
                    f"CHRF={chrf:.3f}, "
                    f"BERT-P={bert_p:.3f}, "
                    f"BERT-R={bert_r:.3f}, "
                    f"BERT-F1={bert_f1:.3f}\n"
                )

if __name__ == "__main__":
    main()
