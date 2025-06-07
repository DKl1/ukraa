import time
import random
import json
import logging
from typing import Dict, List, Tuple, Any
from datasets import load_dataset
from auto_align.aligner import align_sentences
from auto_align.evaluation import evaluate_alignment
from auto_align.encoders.encoder_factory import get_encoder
from auto_align.constants.language_pairs_encoder import LANGUAGE_PAIRS_ENCODER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class FloresEvaluator:
    def __init__(self):
        self.results = {}
        
    def extract_parallel(self, split, src_code: str, tgt_code: str):
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
    
    def test_language_pair(self, src_code: str, tgt_code: str, thresholds: List[float] = None):
        if thresholds is None:
            thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
        
        logger.info(f"Testing {src_code} ⇄ {tgt_code}")
        
        try:
            dev = load_dataset("facebook/flores", "all", split="dev", trust_remote_code=True)
            devtest = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load FLORES dataset: {e}")
            return None
        
        results = {}
        
        for split_name, split in [("dev", dev), ("devtest", devtest)]:
            logger.info(f"Processing {split_name} split ({len(split)} examples)")
            
            try:
                src_sent, tgt_sent = self.extract_parallel(split, src_code, tgt_code)
                
                idx = list(range(len(tgt_sent)))
                random.seed(42)
                random.shuffle(idx)
                tgt_shuffled = [tgt_sent[i] for i in idx]
                
                gold_pairs = set(zip(src_sent, tgt_sent))
                
                split_results = {}
                
                for threshold in thresholds:
                    logger.info(f"  Testing threshold {threshold}")
                    
                    start_time = time.time()
                    
                    aligned = align_sentences(
                        source_sentences=src_sent,
                        target_sentences=tgt_shuffled,
                        src_lang=src_code.split("_")[0].lower(),
                        tgt_lang=tgt_code.split("_")[0].lower(),
                        threshold=threshold
                    )
                    
                    end_time = time.time()
                    
                    metrics = evaluate_alignment(
                        predicted_pairs=aligned,
                        source_sentences=src_sent,
                        target_sentences=tgt_shuffled,
                        gold_pairs=gold_pairs
                    )
                    
                    split_results[threshold] = {
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'f1': metrics.get('f1', 0.0),
                        'num_aligned': len(aligned),
                        'processing_time': end_time - start_time,
                        'sentences_per_second': len(src_sent) / (end_time - start_time)
                    }
                    
                    logger.info(f"    P={metrics.get('precision', 0.0):.3f} R={metrics.get('recall', 0.0):.3f} F1={metrics.get('f1', 0.0):.3f}")
                
                results[split_name] = split_results
                
            except Exception as e:
                logger.error(f"Failed to process {split_name}: {e}")
                continue
        
        return results
    
    def test_encoder_comparison(self, src_lang: str, tgt_lang: str, sample_size: int = 100):
        logger.info(f"Comparing encoders for {src_lang}-{tgt_lang}")
        
        lang_pair = (src_lang, tgt_lang)
        available_encoders = []
        
        for encoder_name in ['labse', 'laser', 'sbert']:
            if lang_pair in LANGUAGE_PAIRS_ENCODER.get(encoder_name, {}):
                available_encoders.append(encoder_name)
        
        if not available_encoders:
            logger.warning(f"No encoders available for {src_lang}-{tgt_lang}")
            return None
        
        test_data = self.generate_test_sentences(src_lang, tgt_lang, sample_size)
        
        results = {}
        for encoder in available_encoders:
            logger.info(f"  Testing {encoder} encoder")
            
            start_time = time.time()
            
            aligned = align_sentences(
                source_sentences=test_data['source'],
                target_sentences=test_data['target_shuffled'],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                threshold=0.7
            )
            
            end_time = time.time()
            
            metrics = evaluate_alignment(
                predicted_pairs=aligned,
                source_sentences=test_data['source'],
                target_sentences=test_data['target_shuffled'],
                gold_pairs=test_data['gold_pairs']
            )
            
            results[encoder] = {
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1', 0.0),
                'processing_time': end_time - start_time
            }
        
        return results
    
    def generate_test_sentences(self, src_lang: str, tgt_lang: str, size: int):
        test_sentences = {
            ('en', 'uk'): [
                ('Hello world', 'Привіт світ'),
                ('How are you?', 'Як справи?'),
                ('Good morning', 'Доброго ранку'),
                ('Thank you very much', 'Дуже дякую'),
                ('See you later', 'До зустрічі'),
                ('What time is it?', 'Котра година?'),
                ('I love you', 'Я тебе кохаю'),
                ('Please help me', 'Будь ласка, допоможи мені'),
                ('Where is the station?', 'Де вокзал?'),
                ('How much does it cost?', 'Скільки це коштує?')
            ],
            ('en', 'de'): [
                ('Hello world', 'Hallo Welt'),
                ('How are you?', 'Wie geht es dir?'),
                ('Good morning', 'Guten Morgen'),
                ('Thank you very much', 'Vielen Dank'),
                ('See you later', 'Bis später'),
                ('What time is it?', 'Wie spät ist es?'),
                ('I love you', 'Ich liebe dich'),
                ('Please help me', 'Bitte hilf mir'),
                ('Where is the station?', 'Wo ist der Bahnhof?'),
                ('How much does it cost?', 'Wie viel kostet das?')
            ],
            ('en', 'fr'): [
                ('Hello world', 'Bonjour le monde'),
                ('How are you?', 'Comment allez-vous?'),
                ('Good morning', 'Bonjour'),
                ('Thank you very much', 'Merci beaucoup'),
                ('See you later', 'À plus tard'),
                ('What time is it?', 'Quelle heure est-il?'),
                ('I love you', 'Je t\'aime'),
                ('Please help me', 'Aidez-moi s\'il vous plaît'),
                ('Where is the station?', 'Où est la gare?'),
                ('How much does it cost?', 'Combien ça coûte?')
            ]
        }
        
        if (src_lang, tgt_lang) in test_sentences:
            pairs = test_sentences[(src_lang, tgt_lang)]
        elif (tgt_lang, src_lang) in test_sentences:
            pairs = [(t, s) for s, t in test_sentences[(tgt_lang, src_lang)]]
        else:
            pairs = [(f"Test sentence {i} in {src_lang}", f"Test sentence {i} in {tgt_lang}") 
                    for i in range(min(size, 20))]
        
        while len(pairs) < size:
            pairs.extend(pairs[:min(len(pairs), size - len(pairs))])
        
        pairs = pairs[:size]
        
        source = [s for s, t in pairs]
        target = [t for s, t in pairs]
        
        idx = list(range(len(target)))
        random.seed(42)
        random.shuffle(idx)
        target_shuffled = [target[i] for i in idx]
        
        return {
            'source': source,
            'target': target,
            'target_shuffled': target_shuffled,
            'gold_pairs': set(pairs)
        }
    
    def run_comprehensive_evaluation(self):
        
        test_pairs = [
            ('ukr_Cyrl', 'eng_Latn'),  
            ('eng_Latn', 'deu_Latn'),  
            ('eng_Latn', 'fra_Latn'),  
            ('deu_Latn', 'fra_Latn'),  
            ('eng_Latn', 'pol_Latn'),  
        ]
        
        all_results = {}
        
        for src_code, tgt_code in test_pairs:
            try:
                results = self.test_language_pair(src_code, tgt_code)
                if results:
                    all_results[f"{src_code}-{tgt_code}"] = results
                    
                    best_f1 = 0
                    best_threshold = 0.7
                    for split in results:
                        for threshold, metrics in results[split].items():
                            if metrics['f1'] > best_f1:
                                best_f1 = metrics['f1']
                                best_threshold = threshold
                    
                    logger.info(f"Best F1 for {src_code}-{tgt_code}: {best_f1:.3f} @ threshold {best_threshold}")
                    
            except Exception as e:
                logger.error(f"Failed to test {src_code}-{tgt_code}: {e}")
                continue
        
        with open('flores_evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results

def main():
    evaluator = FloresEvaluator()
    
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n" + "="*60)
    print("FLORES-200 EVALUATION SUMMARY")
    print("="*60)
    
    for lang_pair, data in results.items():
        print(f"\n{lang_pair}:")
        
        for split in data:
            print(f"  {split}:")
            best_f1 = max(data[split][th]['f1'] for th in data[split])
            best_th = max(data[split], key=lambda th: data[split][th]['f1'])
            print(f"    Best F1: {best_f1:.3f} @ threshold {best_th}")
            
            avg_time = sum(data[split][th]['processing_time'] for th in data[split]) / len(data[split])
            print(f"Avg processing time: {avg_time:.2f}s")

if __name__ == "__main__":
    main() 