import logging
from typing import List, Dict, Tuple
from auto_align.aligner import align_sentences
from auto_align.evaluation import evaluate_alignment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def test_idiomatic_expressions():
    test_pairs = [
        ("It's raining cats and dogs", "Ллє як із відра"),
        ("Break a leg!", "Ні пуху, ні пера!"),
        ("Piece of cake", "Раз плюнути"),
        ("Hit the nail on the head", "Влучити в яблучко"),
        ("Speak of the devil", "Про вовка промовка"),
    ]
    
    results = []
    for en, uk in test_pairs:
        aligned = align_sentences([en], [uk], 'en', 'uk', encoder_name='labse', threshold=0.4)
        score = aligned[0][2] if aligned else 0
        results.append({
            'en': en,
            'uk': uk,
            'similarity': score
        })
    
    return results

def test_cultural_adaptations():
    """Test alignment with cultural adaptations"""
    test_pairs = [
        ("As American as apple pie", "Як вареники з вишнями"),
        ("He graduated from Harvard", "Він закінчив КНУ імені Шевченка"),
        ("They met at Central Park", "Вони зустрілись у Маріїнському парку"),
        ("He loves baseball", "Він любить футбол"),
    ]
    
    results = []
    for en, uk in test_pairs:
        aligned = align_sentences([en], [uk], 'en', 'uk', encoder_name='labse', threshold=0.4)
        score = aligned[0][2] if aligned else 0
        results.append({
            'en': en,
            'uk': uk,
            'similarity': score
        })
    
    return results

def test_paraphrasing():
    original = "The research demonstrates significant results"
    variants = [
        "Дослідження показує важливі результати",
        "З проведеної наукової роботи випливають суттєві висновки",
        "Отримані в ході експерименту дані свідчать про значний прогрес"
    ]
    
    results = []
    for variant in variants:
        aligned = align_sentences([original], [variant], 'en', 'uk', encoder_name='labse', threshold=0.4)
        score = aligned[0][2] if aligned else 0
        results.append({
            'original': original,
            'variant': variant,
            'similarity': score
        })
    
    return results

def test_domain_specific():
    domains = {
        'technical': [
            ("The function returns a boolean value", "Функція повертає логічне значення"),
            ("Click the button to submit the form", "Натисніть кнопку, щоб надіслати форму"),
            ("The API requires authentication", "API вимагає автентифікації")
        ],
        'literary': [
            ("The golden rays of sunset painted the sky", "Золоті промені заходу сонця розфарбували небо"),
            ("Her heart was filled with joy", "Її серце було сповнене радості"),
            ("The ancient castle stood silently", "Старовинний замок стояв мовчки")
        ],
        'legal': [
            ("The parties hereby agree to the following terms", "Сторони цим погоджуються з наступними умовами"),
            ("This agreement shall be governed by applicable law", "Ця угода регулюється чинним законодавством"),
            ("The licensee agrees to indemnify the licensor", "Ліцензіат погоджується відшкодувати збитки ліцензіару")
        ]
    }
    
    results = {}
    for domain, pairs in domains.items():
        domain_results = []
        for en, uk in pairs:
            aligned = align_sentences([en], [uk], 'en', 'uk', encoder_name='labse', threshold=0.6)
            score = aligned[0][2] if aligned else 0
            domain_results.append({
                'en': en,
                'uk': uk,
                'similarity': score
            })
        results[domain] = domain_results
    
    return results

def main():
    logger.info("Starting UKRAA limitations testing...")
    
    # Test idiomatic expressions
    logger.info("\nTesting idiomatic expressions:")
    idiomatic_results = test_idiomatic_expressions()
    for result in idiomatic_results:
        print(f"EN: {result['en']}")
        print(f"UK: {result['uk']}")
        print(f"Similarity: {result['similarity']:.3f}\n")
    
    # Test cultural adaptations
    logger.info("\nTesting cultural adaptations:")
    cultural_results = test_cultural_adaptations()
    for result in cultural_results:
        print(f"EN: {result['en']}")
        print(f"UK: {result['uk']}")
        print(f"Similarity: {result['similarity']:.3f}\n")
    
    # Test paraphrasing
    logger.info("\nTesting paraphrasing:")
    paraphrase_results = test_paraphrasing()
    for result in paraphrase_results:
        print(f"Original: {result['original']}")
        print(f"Variant: {result['variant']}")
        print(f"Similarity: {result['similarity']:.3f}\n")
    
    # Test domain-specific
    logger.info("\nTesting domain-specific alignment:")
    domain_results = test_domain_specific()
    for domain, results in domain_results.items():
        print(f"\n{domain.upper()}:")
        for result in results:
            print(f"EN: {result['en']}")
            print(f"UK: {result['uk']}")
            print(f"Similarity: {result['similarity']:.3f}\n")

if __name__ == "__main__":
    main() 