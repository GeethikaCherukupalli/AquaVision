# -*- coding: utf-8 -*-
"""ResNet18 training script."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ResNet18Gate
from ..preprocessing.normalization import ChannelNormalizer
from ..dataset.index import DatasetIndex
from ..dataset.loader import LazyDataset, PatchDataset
from ..dataset.splitter import SceneSplitter
from ..common.logging import setup_logger, get_logger
from ..common.seed import set_seed
from ..common.validation import validate_image

logger = setup_logger("aquavision.resnet")


class ResNetTrainer:
    """Train ResNet18 candidate gate."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = self.config.get("random_seed", 42)
        set_seed(self.seed)

        # Build model
        self.model = ResNet18Gate(
            input_channels=self.config.get("input_channels", 2),
            pretrained=self.config.get("pretrained", True),
        ).to(self.device)

        # Loss with positive class weighting
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self._get_pos_weight()
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.config.get("scheduler_factor", 0.5),
            patience=self.config.get("scheduler_patience", 2),
        )
        self.mixed_precision = self.config.get("mixed_precision", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        self.epochs = self.config.get("epochs", 10)
        self.batch_size = self.config.get("batch_size", 8)
        self.num_workers = self.config.get("num_workers", 4)
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "artifacts/resnet18"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_pos_weight(self) -> Optional[torch.Tensor]:
        """Calculate positive class weight from training data."""
        weight = self.config.get("positive_class_weight")
        if weight is not None:
            return torch.tensor([float(weight)], device=self.device)
        return None

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        normalizer: Optional[ChannelNormalizer] = None,
    ) -> Dict:
        """Train the ResNet18 gate.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        validation_loader : DataLoader or None
            Validation data loader.
        normalizer : ChannelNormalizer or None
            Normalization configuration.

        Returns
        -------
        dict
            Training history and metrics.
        """
        history = {
            "train": [],
            "validation": [],
            "epoch_metrics": [],
        }
        best_val_loss = float("inf")

        logger.info("Starting ResNet18 training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Mixed precision: {self.mixed_precision}")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []
            epoch_positives = 0
            epoch_total = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(self.device)
                labels = batch["label"].float().to(self.device).unsqueeze(1)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_losses.append(float(loss.item()))
                epoch_positives += int((labels >= 0.5).sum())
                epoch_total += labels.size(0)

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            positive_ratio = (
                epoch_positives / epoch_total if epoch_total > 0 else 0.0
            )

            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"positive_ratio: {positive_ratio:.4f}"
            )

            val_metrics = {}
            if validation_loader is not None:
                val_metrics = self.evaluate(validation_loader)
                self.scheduler.step(val_metrics["loss"])
                history["validation"].append(val_metrics)

                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(self.checkpoint_dir / "best.pth")
            else:
                self.scheduler.step(train_loss)

            history["train"].append(train_loss)
            history["epoch_metrics"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "positive_ratio": positive_ratio,
                    **val_metrics,
                }
            )
            self.save_checkpoint(self.checkpoint_dir / f"epoch_{epoch+1}.pth")

        history["best_val_loss"] = best_val_loss
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        return history

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> Dict:
        """Evaluate the ResNet18 gate.

        Parameters
        ----------
        data_loader : DataLoader
            Evaluation data loader.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        self.model.eval()
        all_logits = []
        all_labels = []

        for batch in data_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probabilities = torch.sigmoid(logits)

        metrics = self._compute_metrics(
            probabilities, labels, self.config.get("classifier_threshold", 0.12)
        )
        metrics["loss"] = float(
            F.binary_cross_entropy_with_logits(logits, labels.float().unsqueeze(1)).item()
        )
        return metrics

    def _compute_metrics(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        threshold: float,
    ) -> Dict:
        """Compute evaluation metrics."""
        predictions = (probabilities >= threshold).int()
        labels_int = labels.int()

        tp = int((predictions * labels_int).sum())
        tn = int(((1 - predictions) * (1 - labels_int)).sum())
        fp = int((predictions * (1 - labels_int)).sum())
        fn = int(((1 - predictions) * labels_int).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        pr_auc = self._average_precision(probabilities, labels_int)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def _average_precision(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute average precision (PR-AUC)."""
        from sklearn.metrics import average_precision_score

        return float(
            average_precision_score(labels.numpy(), probabilities.numpy())
        )

    def threshold_sweep(
        self,
        data_loader: DataLoader,
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Sweep recall targets and identify threshold candidates.

        Parameters
        ----------
        data_loader : DataLoader
            Validation data loader.
        thresholds : list of float or None
            Thresholds to evaluate.

        Returns
        -------
        list of dict
            Threshold evaluation results.
        """
        thresholds = thresholds or [0.01, 0.02, 0.05, 0.1, 0.12, 0.14, 0.2, 0.3, 0.5]
        self.model.eval()
        all_probs = []
        all_labels = []

        for batch in data_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        probabilities = torch.cat(all_probs, dim=0)
        labels = torch.cat(all_labels, dim=0).int()

        results = []
        for threshold in thresholds:
            preds = (probabilities >= threshold).int()
            tp = int((preds * labels).sum())
            fn = int(((1 - preds) * labels).sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fp = int((preds * (1 - labels)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            results.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                }
            )
        return results

    def save_checkpoint(self, path: Path) -> None:
        """Save a model checkpoint with metadata."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: Path) -> None:
        """Load a model checkpoint."""
        checkpoint = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")


def train_resnet(
    dataset_root: str,
    config: Optional[dict] = None,
) -> Dict:
    """Train ResNet18 on the SAR dataset.

    Parameters
    ----------
    dataset_root : str
        Path to the dataset root directory.
    config : dict or None
        Training configuration.

    Returns
    -------
    dict
        Training results including history and artifact paths.
    """
    from .model import ResNet18Gate
    from ..dataset.index import DatasetIndex
    from ..dataset.loader import LazyDataset, PatchDataset
    from ..preprocessing.normalization import ChannelNormalizer, compute_normalization_stats
    from ..preprocessing.patching import PatchExtractor
    from torch.utils.data import DataLoader

    config = config or {}
    dataset = DatasetIndex(dataset_root)
    scenes = dataset.discover_scenes()

    if not scenes:
        raise ValueError(
            f"No scenes found in {dataset_root}. "
            "Check the dataset path and structure."
        )

    # Split scenes and assign labels
    splitter = SceneSplitter(random_seed=config.get("random_seed", 42))
    split = splitter.split(scenes)
    dataset.scenes = split.train + split.validation + split.test

    # Compute normalization statistics from training images only
    logger.info("Computing normalization statistics from training images")
    train_image_paths = [scene.image_path for scene in split.train]
    vv_mean, vv_std, vh_mean, vh_std = compute_normalization_stats(train_image_paths)
    normalizer = ChannelNormalizer(vv_mean, vv_std, vh_mean, vh_std)

    # Build datasets
    train_dataset = LazyDataset(
        dataset_index=dataset,
        normalizer=normalizer,
        split="train",
    )
    val_dataset = LazyDataset(
        dataset_index=dataset,
        normalizer=normalizer,
        split="validation",
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
    )

    # Train model
    trainer = ResNetTrainer(config)
    history = trainer.train(train_loader, val_loader)

    # Save artifacts
    artifact_dir = trainer.checkpoint_dir
    trainer.save_checkpoint(artifact_dir / "resnet18_final.pth")

    # Save metadata
    metadata = {
        "model": "resnet18",
        "input_channels": 2,
        "input_size": [512, 512],
        "channels": ["VV", "VH"],
        "normalization": normalizer.to_dict(),
        "classifier_threshold": config.get("classifier_threshold", 0.12),
        "training_dataset_version": "si2026",
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "history": history,
        "threshold_sweep": trainer.threshold_sweep(val_loader),
    }
    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "history": history,
        "artifact_dir": str(artifact_dir),
        "metadata_path": str(metadata_path),
        "normalizer": normalizer,
    }