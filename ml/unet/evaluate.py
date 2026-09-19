# -*- coding: utf-8 -*-
"""U-Net evaluation and qualitative example generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import UNet
from ..dataset.index import DatasetIndex
from ..dataset.loader import LazyDataset
from ..preprocessing.normalization import ChannelNormalizer
from ..common.logging import setup_logger

logger = setup_logger("aquavision.unet.evaluate")


def evaluate_unet(
    checkpoint_path: str,
    dataset_root: str,
    split: str = "validation",
    batch_size: int = 4,
    output_dir: str | Path | None = None,
) -> Dict:
    """Evaluate a trained U-Net model.

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
    output_dir : str or Path or None
        Directory for qualitative examples.

    Returns
    -------
    dict
        Evaluation results.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model
    model = UNet(pretrained=False).to(device)
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
    all_probs = []
    all_masks = []
    qualitative = []

    for batch in data_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).float().unsqueeze(1)
        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)
        predictions = (probs >= checkpoint.get("config", {}).get(
            "segmentation_threshold", 0.5
        )).float()

        all_probs.append(probs.cpu())
        all_masks.append(masks.cpu())

        if output_dir is not None and len(qualitative) < 10:
            qualitative.append(
                {
                    "scene_id": batch["scene_id"][0],
                    "row": batch["row"][0].item(),
                    "col": batch["col"][0].item(),
                    "image": images[0].cpu().numpy(),
                    "mask": masks[0].cpu().numpy(),
                    "prediction": predictions[0].cpu().numpy(),
                }
            )

    probabilities = torch.cat(all_probs, dim=0)
    masks = torch.cat(all_masks, dim=0)
    predictions = (probabilities >= checkpoint.get("config", {}).get(
        "segmentation_threshold", 0.5
    )).float()

    dice = _dice_score(predictions, masks)
    iou = _iou_score(predictions, masks)
    precision, recall = _precision_recall(predictions, masks)

    results = {
        "checkpoint_path": checkpoint_path,
        "dataset_root": dataset_root,
        "split": split,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "qualitative_examples": qualitative,
    }

    if output_dir is not None:
        save_qualitative_examples(qualitative, output_dir)

    return results


def _dice_score(
    predictions: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compute Dice score."""
    predictions = predictions.view(-1)
    masks = masks.view(-1)
    intersection = (predictions * masks).sum()
    return (2.0 * intersection + 1.0) / (predictions.sum() + masks.sum() + 1.0)


def _iou_score(
    predictions: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compute IoU score."""
    predictions = predictions.view(-1)
    masks = masks.view(-1)
    intersection = (predictions * masks).sum()
    union = predictions.sum() + masks.sum() - intersection
    return (intersection + 1.0) / (union + 1.0)


def _precision_recall(
    predictions: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute precision and recall."""
    predictions = predictions.view(-1)
    masks = masks.view(-1)
    tp = (predictions * masks).sum()
    fp = (predictions * (1 - masks)).sum()
    fn = ((1 - predictions) * masks).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    return precision, recall


def save_qualitative_examples(
    examples: List[dict],
    output_dir: str | Path,
) -> None:
    """Save qualitative examples as JSON metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "qualitative_examples.json", "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "scene_id": example["scene_id"],
                    "row": example["row"],
                    "col": example["col"],
                    "image": example["image"].tolist(),
                    "mask": example["mask"].tolist(),
                    "prediction": example["prediction"].tolist(),
                }
                for example in examples
            ],
            handle,
            indent=2,
        )
    logger.info(f"Saved {len(examples)} qualitative examples to {output_dir}")