# -*- coding: utf-8 -*-
"""U-Net training script."""

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

from .model import UNet
from ..dataset.index import DatasetIndex
from ..dataset.loader import LazyDataset, PatchDataset
from ..preprocessing.normalization import ChannelNormalizer, compute_normalization_stats
from ..common.logging import setup_logger
from ..common.seed import set_seed
from ..common.validation import validate_image, validate_mask

logger = setup_logger("aquavision.unet")


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss."""
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.view(-1)
        targets = targets.view(-1)
        intersection = (probabilities * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probabilities.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEPlusDiceLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class UNetTrainer:
    """Train U-Net for pixel-level segmentation."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = self.config.get("random_seed", 42)
        set_seed(self.seed)

        self.model = UNet(
            input_channels=self.config.get("input_channels", 2),
            output_channels=self.config.get("output_channels", 1),
            base_channels=self.config.get("base_channels", 32),
        ).to(self.device)

        self.criterion = BCEPlusDiceLoss()
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
        self.batch_size = self.config.get("batch_size", 4)
        self.num_workers = self.config.get("num_workers", 4)
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "artifacts/unet"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train": [],
            "validation": [],
            "epoch_metrics": [],
        }

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        normalizer: Optional[ChannelNormalizer] = None,
    ) -> Dict:
        """Train the U-Net model."""
        best_val_loss = float("inf")
        logger.info("Starting U-Net training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device).float().unsqueeze(1)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, masks)
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_losses.append(float(loss.item()))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}"
            )

            val_metrics = {}
            if validation_loader is not None:
                val_metrics = self.evaluate(validation_loader)
                self.scheduler.step(val_metrics["loss"])
                self.history["validation"].append(val_metrics)
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(self.checkpoint_dir / "best.pth")
            else:
                self.scheduler.step(train_loss)

            self.history["train"].append(train_loss)
            self.history["epoch_metrics"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    **val_metrics,
                }
            )
            self.save_checkpoint(self.checkpoint_dir / f"epoch_{epoch+1}.pth")

        self.history["best_val_loss"] = best_val_loss
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        return self.history

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate the U-Net model."""
        self.model.eval()
        all_logits = []
        all_masks = []

        for batch in data_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device).float().unsqueeze(1)
            logits = self.model(images)
            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

        logits = torch.cat(all_logits, dim=0)
        masks = torch.cat(all_masks, dim=0)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.config.get("segmentation_threshold", 0.5)).float()

        dice = self._dice_score(predictions, masks)
        iou = self._iou_score(predictions, masks)
        precision, recall = self._precision_recall(predictions, masks)

        return {
            "loss": float(self.criterion(logits, masks).item()),
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
        }

    def _dice_score(
        self,
        predictions: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice score."""
        predictions = predictions.view(-1)
        masks = masks.view(-1)
        intersection = (predictions * masks).sum()
        return (2.0 * intersection + 1.0) / (predictions.sum() + masks.sum() + 1.0)

    def _iou_score(
        self,
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
        self,
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

    def save_checkpoint(self, path: Path) -> None:
        """Save a model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "epoch": len(self.history.get("train", [])),
            },
            str(path),
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load a model checkpoint."""
        checkpoint = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")


def train_unet(
    dataset_root: str,
    config: Optional[dict] = None,
) -> Dict:
    """Train U-Net on the SAR dataset."""
    from ..dataset.index import DatasetIndex
    from ..dataset.loader import LazyDataset
    from torch.utils.data import DataLoader

    config = config or {}
    dataset = DatasetIndex(dataset_root)
    scenes = dataset.discover_scenes()

    if not scenes:
        raise ValueError(f"No scenes found in {dataset_root}")

    # Split scenes
    from ..dataset.splitter import SceneSplitter

    splitter = SceneSplitter(random_seed=config.get("random_seed", 42))
    split = splitter.split(scenes)
    dataset.scenes = split.train + split.validation + split.test

    # Normalization from training images only
    logger.info("Computing normalization statistics from training images")
    train_image_paths = [scene.image_path for scene in split.train]
    vv_mean, vv_std, vh_mean, vh_std = compute_normalization_stats(train_image_paths)
    normalizer = ChannelNormalizer(vv_mean, vv_std, vh_mean, vh_std)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
    )

    trainer = UNetTrainer(config)
    history = trainer.train(train_loader, val_loader)

    artifact_dir = trainer.checkpoint_dir
    trainer.save_checkpoint(artifact_dir / "unet_final.pth")

    metadata = {
        "model": "unet",
        "input_channels": 2,
        "output_channels": 1,
        "input_size": [512, 512],
        "channels": ["VV", "VH"],
        "normalization": normalizer.to_dict(),
        "segmentation_threshold": config.get("segmentation_threshold", 0.5),
        "training_dataset_version": "si2026",
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "history": history,
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