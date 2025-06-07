import time
import psutil
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
from auto_align.aligner import align_sentences
from auto_align.aligner_no_faiss import align_sentences_no_faiss
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        
    def get_gpu_usage(self) -> Tuple[float, float]:
        if not self.gpu_available:
            return 0.0, 0.0
        
        device = torch.cuda.current_device()
        gpu_util = torch.cuda.utilization(device)
        gpu_mem = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
        return gpu_util, gpu_mem
    
    def get_cpu_usage(self) -> float:
        return self.process.cpu_percent()
    
    def get_ram_usage(self) -> float:
        return self.process.memory_info().rss / 1024**3

def load_test_data(size: int = 1000) -> Tuple[List[str], List[str]]:
    logger.info(f"Generating {size} sample sentence pairs...")
    
    en_samples = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I love programming.",
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
        "Data science is the future.",
        "Natural language processing is amazing.",
        "Artificial intelligence changes everything."
    ]
    
    uk_samples = [
        "Привіт, як справи?",
        "Сьогодні гарна погода.",
        "Я люблю програмування.",
        "Це тестове речення.",
        "Машинне навчання захоплює.",
        "Швидка коричнева лисиця перестрибує через ледачого пса.",
        "Python - чудова мова програмування.",
        "Наука про дані - це майбутнє.",
        "Обробка природної мови вражає.",
        "Штучний інтелект змінює все."
    ]
    
    src_sents = (en_samples * (size // len(en_samples) + 1))[:size]
    tgt_sents = (uk_samples * (size // len(uk_samples) + 1))[:size]
    
    return src_sents, tgt_sents

def test_configuration(
    src_sents: List[str],
    tgt_sents: List[str],
    batch_size: int = 32,
    use_faiss: bool = True
) -> Dict:
    
    monitor = PerformanceMonitor()
    metrics = {
        'batch_size': batch_size,
        'num_sentences': len(src_sents),
        'implementation': 'FAISS' if use_faiss else 'Direct'
    }
    
    align_func = align_sentences if use_faiss else align_sentences_no_faiss
    _ = align_func(
        source_sentences=src_sents[:10],
        target_sentences=tgt_sents[:10],
        src_lang='en',
        tgt_lang='uk',
        batch_size=batch_size
    )
    
    logger.info(f"Testing configuration: batch_size={batch_size}, implementation={'FAISS' if use_faiss else 'Direct'}")
    
    initial_cpu = monitor.get_cpu_usage()
    initial_ram = monitor.get_ram_usage()
    if monitor.gpu_available:
        initial_gpu_util, initial_gpu_mem = monitor.get_gpu_usage()
    
    start_time = time.time()
    
    _ = align_func(
        source_sentences=src_sents,
        target_sentences=tgt_sents,
        src_lang='en',
        tgt_lang='uk',
        batch_size=batch_size
    )
    
    final_cpu = monitor.get_cpu_usage()
    final_ram = monitor.get_ram_usage()
    if monitor.gpu_available:
        final_gpu_util, final_gpu_mem = monitor.get_gpu_usage()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    metrics.update({
        'sentences_per_second': len(src_sents) / processing_time,
        'total_time_seconds': processing_time,
        'avg_cpu_usage': (initial_cpu + final_cpu) / 2,
        'max_cpu_usage': max(initial_cpu, final_cpu),
        'avg_ram_gb': (initial_ram + final_ram) / 2,
        'max_ram_gb': max(initial_ram, final_ram)
    })
    
    if monitor.gpu_available:
        metrics.update({
            'avg_gpu_util': (initial_gpu_util + final_gpu_util) / 2,
            'max_gpu_util': max(initial_gpu_util, final_gpu_util),
            'avg_gpu_mem_gb': (initial_gpu_mem + final_gpu_mem) / 2,
            'max_gpu_mem_gb': max(initial_gpu_mem, final_gpu_mem)
        })
    
    return metrics

def main():
    configs = [
        {'batch_size': 32, 'use_faiss': True},   # FAISS standard batch
        {'batch_size': 64, 'use_faiss': True},   # FAISS large batch
        {'batch_size': 16, 'use_faiss': True},   # FAISS small batch
        {'batch_size': 32, 'use_faiss': False},  # Direct standard batch
        {'batch_size': 64, 'use_faiss': False},  # Direct large batch
        {'batch_size': 16, 'use_faiss': False},  # Direct small batch
    ]
    
    sizes = [100, 500, 1000]
    
    results = []
    
    for size in sizes:
        logger.info(f"\nTesting with {size} sentences")
        src_sents, tgt_sents = load_test_data(size)
        
        for config in configs:
            try:
                metrics = test_configuration(
                    src_sents=src_sents,
                    tgt_sents=tgt_sents,
                    **config
                )
                metrics['test_size'] = size
                results.append(metrics)
                
                logger.info(f"Results for {config}:")
                logger.info(f"  Speed: {metrics['sentences_per_second']:.1f} sent/sec")
                logger.info(f"  RAM: {metrics['max_ram_gb']:.1f} GB")
                if 'max_gpu_mem_gb' in metrics:
                    logger.info(f"  GPU Memory: {metrics['max_gpu_mem_gb']:.1f} GB")
                
            except Exception as e:
                logger.error(f"Error testing configuration {config}: {str(e)}")
    
    import json
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 