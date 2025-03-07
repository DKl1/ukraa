"""
Module for evaluation metrics.

This module provides a function to compute metrics for text alignment
such as precision@1, precision@k, and mean reciprocal rank (MRR).
"""


import numpy as np
from typing import Dict, List


def compute_metrics(pred_indices: np.ndarray, gold: np.ndarray, metrics: List[str] = None, k: int = 1) -> Dict[
    str, float]:
    """
    Compute evaluation metrics based on predicted indices and gold standard indices.

    Args:
        pred_indices (np.ndarray): 2D array of predicted English indices for each Spanish sentence.
        gold (np.ndarray): 1D array of gold standard indices.
        metrics (List[str], optional): List of metrics to compute. Defaults to ["precision@1", "precision@k", "mrr"].
        k (int, optional): Number of top predictions to consider for precision@k. Defaults to 1.

    Returns:
        Dict[str, float]: Dictionary with metric names as keys and computed metric values as values.
    """

    if metrics is None:
        metrics = ["precision@1", "precision@k", "mrr"]
    num_samples = gold.shape[0]
    correct_top1 = 0
    correct_in_topk = 0
    reciprocal_ranks = []

    for i in range(num_samples):
        preds = pred_indices[i]
        gold_idx = gold[i]
        if preds[0] == gold_idx:
            correct_top1 += 1
        if gold_idx in preds:
            correct_in_topk += 1
            rank = np.where(preds == gold_idx)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    results = {}
    if "precision@1" in metrics:
        results["precision@1"] = correct_top1 / num_samples
    if "precision@k" in metrics:
        results[f"precision@{k}"] = correct_in_topk / num_samples
    if "mrr" in metrics:
        results["mrr"] = np.mean(reciprocal_ranks)

    return results
