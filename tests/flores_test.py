import random
import argparse
import logging

from datasets import load_dataset
from auto_align.aligner import align_sentences     
from auto_align.evaluation import evaluate_alignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_parallel(split, src_code: str, tgt_code: str):

    if not split:
        return [], []
    example = split[0]
    keys = set(example.keys())

    def resolve_key(want: str) -> str:
        if want in keys:
            return want
        low = want.lower()
        for k in keys:
            if k.lower() == low:
                return k
        pref = f"sentence_{want}"
        if pref in keys:
            return pref
        low2 = pref.lower()
        for k in keys:
            if k.lower() == low2:
                return k
        raise KeyError(f"Could not find column for '{want}'. Available keys: {list(keys)[:5]}…")

    use_trans = "translation" in example

    if not use_trans:
        real_src = resolve_key(src_code)
        real_tgt = resolve_key(tgt_code)

    src_list, tgt_list = [], []
    for ex in split:
        if use_trans:
            src_list.append(ex["translation"][src_code])
            tgt_list.append(ex["translation"][tgt_code])
        else:
            src_list.append(ex[real_src])
            tgt_list.append(ex[real_tgt])
    return src_list, tgt_list

def run_flores_eval(src_code: str, tgt_code: str, threshold: float):
    logger.info(f"Loading FLORES-200 dev/devtest for {src_code} ⇄ {tgt_code}")
    dev = load_dataset("facebook/flores", "all", split="dev", trust_remote_code=True)
    devtest = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)
    print(dev.column_names)
    print(dev.features)

    for split_name, split in [("dev", dev), ("devtest", devtest)]:
        logger.info(f"  — split={split_name}, examples={len(split)}")
        src_sent, tgt_sent = extract_parallel(split, src_code, tgt_code)

        idx = list(range(len(tgt_sent)))
        random.seed(42)
        random.shuffle(idx)
        tgt_shuffled = [tgt_sent[i] for i in idx]

        gold_pairs = set(zip(src_sent, tgt_sent))

        logger.info(f"  Aligning (threshold={threshold})…")
        aligned = align_sentences(
            source_sentences=src_sent,
            target_sentences=tgt_shuffled,
            src_lang=src_code.split("_")[0].lower(),  
            tgt_lang=tgt_code.split("_")[0].lower(),   
            threshold=threshold
        )

        metrics = evaluate_alignment(
            predicted_pairs=aligned,
            source_sentences=src_sent,
            target_sentences=tgt_shuffled,
            gold_pairs=gold_pairs
        )
        
        prec = metrics.get('precision', 0.0)
        rec = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        
        logger.info(f"  {split_name}: Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FLORES-200 evaluation")
    p.add_argument("--src",  required=True, help="FLORES code for source (e.g. ukr_Cyrl)")
    p.add_argument("--tgt",  required=True, help="FLORES code for target (e.g. eng_Latn)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="cosine threshold for alignment")
    args = p.parse_args()

    run_flores_eval(src_code=args.src, tgt_code=args.tgt, threshold=args.threshold)
