import time
import json
from auto_align.aligner import align_sentences
from auto_align.encoders.encoder_factory import get_encoder

def run_performance_test():
    """Test alignment performance with different configurations"""
    
    test_sets = {
        'small': {
            'en': ['Hello world', 'How are you?', 'Good morning', 'Thank you'],
            'uk': ['Привіт світ', 'Як справи?', 'Доброго ранку', 'Дякую тобі']
        },
        'medium': {
            'en': ['Hello world', 'How are you?', 'Good morning', 'Thank you', 'Goodbye', 
                   'Please help me', 'What time is it?', 'I love you', 'See you later', 'Welcome'],
            'uk': ['Привіт світ', 'Як справи?', 'Доброго ранку', 'Дякую тобі', 'До побачення',
                   'Будь ласка допоможи мені', 'Котра година?', 'Я тебе кохаю', 'Побачимось пізніше', 'Ласкаво просимо']
        }
    }
    
    thresholds = [0.3, 0.5, 0.7, 0.8]
    
    results = {
        'algorithm_overview': {
            'name': 'UKRAA - Ukrainian Auto Alignment',
            'approach': 'Multilingual embedding-based semantic similarity',
            'encoders': ['LaBSE', 'LASER', 'SBERT'],
            'similarity_metric': 'Cosine similarity',
            'search_method': 'FAISS approximate nearest neighbors'
        },
        'encoder_tests': {},
        'threshold_analysis': {},
        'performance_metrics': {}
    }
    
    print("=== UKRAA Evaluation ===\n")
    
    encoders = ['labse', 'sbert', 'laser']
    
    for encoder_name in encoders:
        print(f"Testing encoder: {encoder_name.upper()}")
        
        try:
            start_time = time.time()
            alignments = align_sentences(
                test_sets['medium']['en'], 
                test_sets['medium']['uk'],
                'en', 'uk', 
                encoder_name=encoder_name,
                threshold=0.7
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            results['encoder_tests'][encoder_name] = {
                'alignments_found': len(alignments),
                'processing_time_seconds': round(processing_time, 4),
                'top_alignments': [(s, t, round(score, 3)) for s, t, score in alignments[:5]]
            }
            
            print(f" {len(alignments)} alignments found in {processing_time:.3f}s")
            
        except Exception as e:
            print(f"Error with {encoder_name}: {str(e)}")
            results['encoder_tests'][encoder_name] = {'error': str(e)}
    
    print(f"\nTesting threshold sensitivity with automatic encoder...")
    
    for threshold in thresholds:
        start_time = time.time()
        alignments = align_sentences(
            test_sets['medium']['en'], 
            test_sets['medium']['uk'],
            'en', 'uk',
            threshold=threshold
        )
        end_time = time.time()
        
        results['threshold_analysis'][threshold] = {
            'alignments_found': len(alignments),
            'processing_time_seconds': round(end_time - start_time, 4),
            'precision_estimate': calculate_precision_estimate(alignments, threshold)
        }
        
        print(f"  Threshold {threshold}: {len(alignments)} alignments")
    
    print(f"\nTesting scalability...")
    
    large_en = test_sets['medium']['en'] * 10  # 100 sentences
    large_uk = test_sets['medium']['uk'] * 10  # 100 sentences
    
    start_time = time.time()
    large_alignments = align_sentences(large_en, large_uk, 'en', 'uk', threshold=0.7)
    end_time = time.time()
    
    results['performance_metrics'] = {
        'large_corpus_test': {
            'source_sentences': len(large_en),
            'target_sentences': len(large_uk),
            'alignments_found': len(large_alignments),
            'processing_time_seconds': round(end_time - start_time, 4),
            'sentences_per_second': round((len(large_en) + len(large_uk)) / (end_time - start_time), 2)
        }
    }
    
    print(f"Large corpus: {len(large_alignments)} alignments for {len(large_en)}×{len(large_uk)} in {end_time - start_time:.3f}s")
    
    return results

def calculate_precision_estimate(alignments, threshold):
    if not alignments:
        return 0.0
    
    high_confidence = sum(1 for _, _, score in alignments if score > threshold + 0.1)
    return round(high_confidence / len(alignments), 3)

def test_language_pairs():
    
    pairs = [
        ('en', 'uk', 'English-Ukrainian'),
        ('en', 'es', 'English-Spanish'), 
        ('uk', 'pl', 'Ukrainian-Polish'),
        ('en', 'zh', 'English-Chinese')
    ]
    
    test_en = ['Hello', 'Thank you', 'Good morning']
    test_sentences = {
        'uk': ['Привіт', 'Дякую', 'Доброго ранку'],
        'es': ['Hola', 'Gracias', 'Buenos días'],
        'pl': ['Cześć', 'Dziękuję', 'Dzień dobry'],
        'zh': ['你好', '谢谢', '早上好']
    }
    
    results = {}
    
    print("=== Language Pair Testing ===\n")
    
    for src, tgt, name in pairs:
        print(f"Testing {name} ({src}-{tgt})")
        
        try:
            src_sentences = test_en if src == 'en' else test_sentences[src]
            tgt_sentences = test_en if tgt == 'en' else test_sentences[tgt]
            
            # Get encoder info
            encoder = get_encoder(languages=(src, tgt))
            encoder_type = type(encoder).__name__
            
            alignments = align_sentences(src_sentences, tgt_sentences, src, tgt, threshold=0.5)
            
            results[f"{src}-{tgt}"] = {
                'language_pair': name,
                'encoder_used': encoder_type,
                'alignments': len(alignments),
                'sample_alignment': alignments[0] if alignments else None
            }
            
            print(f"Encoder: {encoder_type}, Alignments: {len(alignments)}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results[f"{src}-{tgt}"] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("Starting UKRAA evaluation...\n")
    
    performance_results = run_performance_test()
    
    print("\n" + "="*50 + "\n")
    
    language_results = test_language_pairs()
    
    full_results = {
        'performance_evaluation': performance_results,
        'language_pair_evaluation': language_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('ukraa_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    print(f"Summary:")
    print(f"Encoders tested: {len(performance_results['encoder_tests'])}")
    print(f"Thresholds tested: {len(performance_results['threshold_analysis'])}")
    print(f"Language pairs tested: {len(language_results)}") 