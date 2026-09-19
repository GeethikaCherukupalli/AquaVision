# -*- coding: utf-8 -*-
"""ResNet18 evaluation and threshold analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ResNet18Gate
from ..dataset.index import DatasetIndex
from ..dataset.loader import LazyDataset, PatchDataset
from ..preprocessing.normalization import ChannelNormalizer
from ..common.logging import setup_logger

logger = setup_logger("aquavision.resnet.evaluate")


def evaluate_resnet(
    checkpoint_path: str,
    dataset_root: str,
    split: str = "validation",
    batch_size: int = 8,
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Evaluate a trained ResNet18 gate.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint.
    dataset_root : str
        Path to the dataset root directory.
    split : str
        Dataset split to evaluate.
    batch_size : int
        Batch size for evaluation.
    thresholds : list of float or None
        Thresholds for the sweep.

    Returns
    -------
    dict
        Evaluation results with metrics per threshold.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model
    model = ResNet18Gate(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset
    dataset_index = DatasetIndex(dataset_root)
    scenes = dataset_index.discover_scenes()
    normalizer = ChannelNormalizer(
        vv_mean=checkpoint.get("config", {}).get(
            "normalization", {}
        ).get("vv_mean"),
        vv_std=checkpoint.get("config", {}).get(
            "normalization", {}
        ).get("vv_std"),
        vh_mean=checkpoint.get("config", {}).get(
            "normalization", {}
        ).get("vh_mean"),
        vh_std=checkpoint.get("config", {}).get(
            "normalization", {}
        ).get("vh_std"),
    )

    lazy_dataset = LazyDataset(
        dataset_index=dataset_index,
        normalizer=normalizer,
        split=split,
    )
    data_loader = DataLoader(
        lazy_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Run evaluation
    thresholds = thresholds or [0.01, 0.02, 0.05, 0.1, 0.12, 0.14, 0.2, 0.3, 0.5]
    all_probs = []
    all_labels = []

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probabilities = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0).int()

    # Threshold sweep
    results = []
    for threshold in thresholds:
        preds = (probabilities >= threshold).int()
        tp = int((preds * labels).sum())
        tn = int(((1 - preds) * (1 - labels)).sum())
        fp = int((preds * (1 - labels)).sum())
        fn = int(((1 - preds) * labels).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        pr_auc = float(
            F.binary_cross_entropy_with_logits(probabilities, labels.float()).item()
        )
        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pr_auc": pr_auc,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    return {
        "checkpoint_path": checkpoint_path,
        "dataset_root": dataset_root,
        "split": split,
        "thresholds": results,
        "best_threshold": max(results, key=lambda x: x["f1"]),
    }


def save_evaluation_report(
    results: Dict,
    path: str | Path,
) -> None:
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Saved evaluation report to {path}")