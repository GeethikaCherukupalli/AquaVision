# -*- coding: utf-8 -*-
"""Evaluation metrics for binary classification and segmentation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute binary classification metrics."""
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0,
        "pr_auc": float(_average_precision(probabilities, labels)),
    }


def segmentation_metrics(
    predictions: np.ndarray,
    masks: np.ndarray,
) -> Dict:
    """Compute segmentation metrics."""
    predictions = np.asarray(predictions, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.int64)

    tp = int(np.sum((predictions == 1) & (masks == 1)))
    tn = int(np.sum((predictions == 0) & (masks == 0)))
    fp = int(np.sum((predictions == 1) & (masks == 0)))
    fn = int(np.sum((predictions == 0) & (masks == 1)))

    dice = (2 * tp + 1.0) / (2 * tp + fp + fn + 1.0)
    iou = (tp + 1.0) / (tp + fp + fn + 1.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_sweep(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[Dict]:
    """Sweep classification thresholds."""
    thresholds = thresholds or [0.01, 0.02, 0.05, 0.1, 0.12, 0.14, 0.2, 0.3, 0.5]
    results = []

    for threshold in thresholds:
        metrics = binary_metrics(probabilities, labels, threshold)
        results.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    return results


def _average_precision(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute average precision (PR-AUC)."""
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(labels, probabilities))
    except ImportError:
        # Fallback implementation
        order = np.argsort(probabilities)[::-1]
        sorted_labels = labels[order]
        positives = np.sum(sorted_labels)
        if positives == 0:
            return 0.0
        precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
        return float(np.sum(precision * sorted_labels) / positives)