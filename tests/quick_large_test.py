import time
import random
import json
from typing import List, Tuple
from auto_align.aligner import align_sentences

def generate_corpus(size: int, src_lang: str = 'uk', tgt_lang: str = 'en') -> Tuple[List[str], List[str]]:
    
    templates = {
        'uk': [
            "Це речення номер {} в українському корпусі.",
            "Система автоматичного вирівнювання працює з реченням {}.",
            "Дослідження багатомовних технологій включає приклад {}.",
            "Українська мова є важливою для тестування номер {}.",
            "Комп'ютерна лінгвістика розвивається завдяки дослідженням {}.",
            "Машинний переклад покращується через експеримент {}.",
            "Семантичні вкладення допомагають у завданні {}.",
            "Штучний інтелект обробляє текст під номером {}.",
            "Природна мова моделюється в прикладі {}.",
            "Багатомовність важлива для дослідження {}."
        ],
        'en': [
            "This is sentence number {} in the English corpus.",
            "Automatic alignment system processes sentence {}.",
            "Multilingual technology research includes example {}.",
            "English language is important for testing number {}.",
            "Computational linguistics advances through research {}.",
            "Machine translation improves via experiment {}.",
            "Semantic embeddings assist in task {}.",
            "Artificial intelligence processes text number {}.",
            "Natural language is modeled in example {}.",
            "Multilinguality matters for research {}."
        ],
        'de': [
            "Das ist Satz Nummer {} im deutschen Korpus.",
            "Automatisches Alignment-System verarbeitet Satz {}.",
            "Mehrsprachige Technologieforschung umfasst Beispiel {}.",
            "Deutsche Sprache ist wichtig für Test Nummer {}.",
            "Computerlinguistik entwickelt sich durch Forschung {}.",
            "Maschinelle Übersetzung verbessert sich durch Experiment {}.",
            "Semantische Einbettungen helfen bei Aufgabe {}.",
            "Künstliche Intelligenz verarbeitet Text Nummer {}.",
            "Natürliche Sprache wird in Beispiel {} modelliert.",
            "Mehrsprachigkeit ist wichtig für Forschung {}."
        ],
        'fr': [
            "Ceci est la phrase numéro {} dans le corpus français.",
            "Le système d'alignement automatique traite la phrase {}.",
            "La recherche technologique multilingue inclut l'exemple {}.",
            "La langue française est importante pour le test numéro {}.",
            "La linguistique computationnelle progresse grâce à la recherche {}.",
            "La traduction automatique s'améliore via l'expérience {}.",
            "Les plongements sémantiques aident dans la tâche {}.",
            "L'intelligence artificielle traite le texte numéro {}.",
            "Le langage naturel est modélisé dans l'exemple {}.",
            "Le multilinguisme compte pour la recherche {}."
        ],
        'pl': [
            "To jest zdanie numer {} w polskim korpusie.",
            "System automatycznego wyrównania przetwarza zdanie {}.",
            "Badania technologii wielojęzycznych obejmują przykład {}.",
            "Język polski jest ważny dla testu numer {}.",
            "Lingwistyka komputerowa rozwija się poprzez badania {}.",
            "Tłumaczenie maszynowe poprawia się dzięki eksperymentowi {}.",
            "Osadzenia semantyczne pomagają w zadaniu {}.",
            "Sztuczna inteligencja przetwarza tekst numer {}.",
            "Język naturalny jest modelowany w przykładzie {}.",
            "Wielojęzyczność ma znaczenie dla badań {}."
        ]
    }
    
    src_templates = templates.get(src_lang, templates['uk'])
    tgt_templates = templates.get(tgt_lang, templates['en'])
    
    src_sentences = []
    tgt_sentences = []
    
    for i in range(size):
        src_template = random.choice(src_templates)
        tgt_template = random.choice(tgt_templates)
        
        src_sentences.append(src_template.format(i + 1))
        tgt_sentences.append(tgt_template.format(i + 1))
    
    return src_sentences, tgt_sentences

def test_ukrainian_pairs():
    
    language_pairs = [
        ('uk', 'en', 'Ukrainian-English'),
        ('uk', 'de', 'Ukrainian-German'), 
        ('uk', 'fr', 'Ukrainian-French'),
        ('uk', 'pl', 'Ukrainian-Polish'),
        ('en', 'uk', 'English-Ukrainian'),
        ('de', 'uk', 'German-Ukrainian'),
        ('fr', 'uk', 'French-Ukrainian'),
        ('pl', 'uk', 'Polish-Ukrainian')
    ]
    
    corpus_sizes = [5000, 10000, 15000, 20000]
    
    results = {}
    
    for src_lang, tgt_lang, pair_name in language_pairs:
        print(f"Testing {pair_name}")
        pair_results = {}
        
        for size in corpus_sizes:
            print(f"  Testing with {size} sentences")
            
            try:
                src_sentences, tgt_sentences = generate_corpus(size, src_lang, tgt_lang)
                
                tgt_shuffled = tgt_sentences.copy()
                random.seed(42)
                random.shuffle(tgt_shuffled)
                
                start_time = time.time()
                
                aligned = align_sentences(
                    source_sentences=src_sentences,
                    target_sentences=tgt_shuffled,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    threshold=0.7
                )
                
                end_time = time.time()
                
                processing_time = end_time - start_time
                sentences_per_second = size / processing_time if processing_time > 0 else 0
                
                pair_results[size] = {
                    'processing_time': processing_time,
                    'sentences_per_second': sentences_per_second,
                    'num_alignments': len(aligned),
                    'alignment_rate': len(aligned) / size,
                    'success': True
                }
                
                print(f"    {size} sentences: {processing_time:.2f}s, "
                      f"{sentences_per_second:.1f} sent/s, "
                      f"{len(aligned)} alignments")
                
            except Exception as e:
                print(f"    Failed {size} sentences: {e}")
                pair_results[size] = {
                    'error': str(e),
                    'success': False
                }
                continue
        
        results[pair_name] = pair_results
    
    return results

def main():
    print("Starting large-scale Ukrainian-focused evaluation")
    
    results = test_ukrainian_pairs()

    with open('quick_large_scale_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("LARGE-SCALE EVALUATION SUMMARY")
    print("="*60)
    
    for pair_name, pair_data in results.items():
        if isinstance(pair_data, dict):
            max_successful = 0
            best_rate = 0
            for size, metrics in pair_data.items():
                if isinstance(metrics, dict) and metrics.get('success', False):
                    if size > max_successful:
                        max_successful = size
                        best_rate = metrics['sentences_per_second']
            
            if max_successful > 0:
                print(f"{pair_name}: {max_successful:,} sentences @ {best_rate:.1f} sent/s")

if __name__ == "__main__":
    main() 