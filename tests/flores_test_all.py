import argparse
import logging
import random
from datasets import load_dataset

from auto_align.evaluation import evaluate_alignment
from auto_align.aligner import align_sentences
from auto_align.constants.language_pairs_encoder import PREFERRED_ENCODER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLORES_CODE = {
    "uk": "ukr_Cyrl",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "pl": "pol_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "es": "spa_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "hr": "hrv_Latn",
    "bg": "bul_Cyrl",
    "sr": "srp_Cyrl",
    "tr": "tur_Latn",
    "pt": "por_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "nb": "nob_Latn",
    "fi": "fin_Latn",
    "et": "est_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "vi": "vie_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "fa": "prs_Arab",
    "ka": "kat_Geor",
    "az": "azj_Latn",
    "he": "heb_Hebr",
    "uz": "uzn_Latn",
    "kk": "kaz_Cyrl",
    "ms": "zsm_Latn",
    "id": "ind_Latn",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "af": "afr_Latn",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
    "th": "tha_Thai",
}

def extract_parallel(split, src_code: str, tgt_code: str):

    if not split:
        return [], []
    example = split[0]
    keys = set(example.keys())

    def resolve_key(lang: str) -> str:
        if lang in keys:
            return lang
        low = lang.lower()
        for k in keys:
            if k.lower() == low:
                return k
        pref = f"sentence_{lang}"
        if pref in keys:
            return pref
        lowp = pref.lower()
        for k in keys:
            if k.lower() == lowp:
                return k
        raise KeyError(f"Cannot find column for '{lang}'")

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

def run_one_pair(src_flores: str, tgt_flores: str, threshold: float):
    dev   = load_dataset("facebook/flores", "all", split="dev",   trust_remote_code=True)
    devt  = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)
    for split_name, split in [("dev", dev), ("devtest", devt)]:
        logger.info(f"▶️  {src_flores} ⇄ {tgt_flores} — {split_name} ({len(split)} examples)")
        src_sents, tgt_sents = extract_parallel(split, src_flores, tgt_flores)

        # shuffle target, keep gold mapping
        idx = list(range(len(tgt_sents)))
        random.seed(42)
        random.shuffle(idx)
        tgt_shuffled = [tgt_sents[i] for i in idx]
        gold_pairs   = set(zip(src_sents, tgt_sents))

        # align
        aligned = align_sentences(
            source_sentences=src_sents,
            target_sentences=tgt_shuffled,
            src_lang=src_flores.split("_")[0].lower(),
            tgt_lang=tgt_flores.split("_")[0].lower(),
            threshold=threshold
        )

        # eval
        metrics = evaluate_alignment(
            predicted_pairs=aligned,
            source_sentences=src_sents,
            target_sentences=tgt_shuffled,
            gold_pairs=gold_pairs
        )
        logger.info(f"Evaluation metrics for {split_name} ({src_flores} → {tgt_flores}):")
        for name, value in metrics.items():
            logger.info(f"    {name.upper():<20} = {value:.4f}")
def main():
    p = argparse.ArgumentParser(description="FLORES‐200 all-pairs eval")
    p.add_argument("--threshold", "-th", type=float, default=0.7, help="cosine threshold")
    args = p.parse_args()

    seen = set()
    for (a,b), model in PREFERRED_ENCODER.items():
        if a!="uk":
            continue
        if b in seen:
            continue
        seen.add(b)
        
        if b not in FLORES_CODE:
            logger.warning(f"Skipping language '{b}' - not found in FLORES_CODE dictionary")
            continue
            
        src_f = FLORES_CODE["uk"]
        tgt_f = FLORES_CODE[b]
        logger.info(f"\n===== Testing pair: uk -> {b} ({model}) =====")
        try:
            run_one_pair(src_f, tgt_f, args.threshold)
        except Exception as e:
            logger.error(f"Error testing pair uk -> {b}: {e}")
            continue

if __name__=="__main__":
    main()
