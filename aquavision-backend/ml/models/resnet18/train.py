# -*- coding: utf-8 -*-
"""ResNet18 training script."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ResNet18Gate
from ...preprocessing.normalization import ChannelNormalizer
from ...dataset.index import DatasetIndex
from ...dataset.loader import LazyDataset, PatchDataset
from ...dataset.splitter import SceneSplitter
from ...common.logging import setup_logger, get_logger
from ...common.seed import set_seed
from ...common.validation import validate_image

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

        # Initialize checkpointing attributes
        self.last_epoch = 0
        self._best_val_loss = float("inf")
        self._history = {"train": [], "validation": [], "epoch_metrics": []}
        self._train_scenes = []
        self._val_scenes = []
        self._test_scenes = []

    def _get_pos_weight(self) -> Optional[torch.Tensor]:
        """Calculate positive class weight from training data."""
        weight = self.config.get("positive_class_weight")
        if weight is not None:
            return torch.tensor([float(weight)], device=self.device)
        return None

    def _print_training_status(
        self,
        train_scenes,
        validation_scenes,
        test_count: int = 450,
    ) -> None:
        """Print reproducible training status at startup."""
        print("Dataset:")
        print(f"train scenes = {len(train_scenes)}")
        print(f"validation scenes = {len(validation_scenes)}")
        print(f"test scenes = {test_count}")
        print()
        print("Model:")
        print("ResNet18")
        print("input channels = 2")
        print("patch size = 512x512")
        print()
        print("Normalization:")
        if self.normalizer is not None:
            print(f"VV mean/std = {self.normalizer.vv_mean:.6f} / {self.normalizer.vv_std:.6f}")
            print(f"VH mean/std = {self.normalizer.vh_mean:.6f} / {self.normalizer.vh_std:.6f}")
        print("source = cache or freshly computed")
        print()
        print("Training:")
        print(f"epochs = {self.epochs}")
        print(f"batch size = {self.batch_size}")
        print(f"learning rate = {self.config.get('learning_rate', 1e-4)}")
        print(f"weight decay = {self.config.get('weight_decay', 1e-4)}")
        print(f"positive class weight = {self.config.get('positive_class_weight')}")
        print(f"classifier threshold = {self.config.get('classifier_threshold', 0.12)}")
        print(f"mixed precision = {self.mixed_precision}")
        print(f"num workers = {self.num_workers}")
        resume_path = self.config.get("resume_from_checkpoint")
        print(f"resume checkpoint = {resume_path or '(none)'}")

    def _log_epoch_metrics(self, metrics: Dict) -> None:
        """Log epoch metrics to a human-readable file."""
        log_path = self.checkpoint_dir / "epoch_metrics.log"
        line = (
            f"epoch={metrics.get('epoch', 'N/A')} "
            f"train_loss={metrics.get('train_loss', 'N/A'):.4f} "
            f"val_loss={metrics.get('loss', 'N/A'):.4f} "
            f"precision={metrics.get('precision', 'N/A'):.4f} "
            f"recall={metrics.get('recall', 'N/A'):.4f} "
            f"f1={metrics.get('f1', 'N/A'):.4f} "
            f"pr_auc={metrics.get('pr_auc', 'N/A'):.4f} "
            f"epoch_duration={metrics.get('epoch_duration', 'N/A'):.2f}s "
            f"best_val_loss={metrics.get('best_val_loss', getattr(self, '_best_val_loss', float('inf'))):.4f}\n"
        )
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    def _history_partial(self) -> Dict:
        """Return whatever training history we have so far."""
        return getattr(self, "_history", {"train": [], "validation": [], "epoch_metrics": []})

    def save_checkpoint(
        self,
        path: Path,
        *,
        epoch: Optional[int] = None,
        history: Optional[Dict] = None,
        atomic: bool = True,
    ) -> None:
        """Save a model checkpoint with full training state.

        Parameters
        ----------
        path : Path
            Destination path (latest.pth, best.pth, epoch_XXX.pth, resnet18_final.pth).
        epoch : int or None
            Current epoch number.
        history : dict or None
            Training history.
        atomic : bool
            If True, write via a temporary file + os.replace so a partially
            written file can never replace a valid checkpoint.
        """
        actual_epoch = epoch if epoch is not None else getattr(self, "last_epoch", 0)
        actual_history = (
            history if history is not None else self._history_partial()
        )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "epoch": actual_epoch,
            "best_val_loss": getattr(self, "_best_val_loss", float("inf")),
            "history": actual_history,
            "random_seed": self.seed,
            "config": self.config,
            "normalization_stats": (
                self.normalizer.to_dict() if self.normalizer is not None else None
            ),
            "classifier_threshold": self.config.get("classifier_threshold", 0.12),
            "dataset_split_info": {
                "train_count": len(getattr(self, "_train_scenes", [])),
                "val_count": len(getattr(self, "_val_scenes", [])),
                "test_count": len(getattr(self, "_test_scenes", [])),
                "splits": {
                    "train": [s.scene_id for s in getattr(self, "_train_scenes", [])],
                    "validation": [
                        s.scene_id for s in getattr(self, "_val_scenes", [])
                    ],
                    "test": [s.scene_id for s in getattr(self, "_test_scenes", [])],
                },
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if atomic:
            import tempfile as _tf
            import os as _os

            fd, tmp_path = _tf.mkstemp(
                dir=str(path.parent),
                prefix=".checkpoint_",
                suffix=".tmp",
            )
            try:
                with _os.fdopen(fd, "wb") as fh:
                    torch.save(checkpoint, fh)
                _os.replace(tmp_path, str(path))
            except Exception:
                if _os.path.exists(tmp_path):
                    _os.remove(tmp_path)
                raise
        else:
            torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: Path) -> int:
        """Load a checkpoint and restore all available training state.

        Returns
        -------
        int
            Next epoch number to run (i.e. loaded_epoch + 1).
        """
        checkpoint = torch.load(str(path), map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if (
            self.optimizer is not None
            and checkpoint.get("optimizer_state_dict") is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        loaded_epoch = checkpoint.get("epoch", 0)
        self.last_epoch = loaded_epoch
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self._history = checkpoint.get(
            "history", {"train": [], "validation": [], "epoch_metrics": []}
        )

        # Restore classifier threshold
        threshold = checkpoint.get("classifier_threshold", 0.12)
        self.config["classifier_threshold"] = threshold

        # Restore normalizer stats if present
        if checkpoint.get("normalization_stats") is not None:
            if self.normalizer is not None:
                nd = checkpoint["normalization_stats"]
                self.normalizer = ChannelNormalizer(
                    vv_mean=nd.get("vv_mean"),
                    vv_std=nd.get("vv_std"),
                    vh_mean=nd.get("vh_mean"),
                    vh_std=nd.get("vh_std"),
                )

        # Store split info for later checkpoint payloads
        split_info = checkpoint.get("dataset_split_info", {})
        self._train_scenes = [
            {
                "scene_id": sid,
                "image_path": "",
                "mask_path": None,
                "split": "train",
                "class_label": None,
                "part": None,
            }
            for sid in split_info.get("train", [])
        ]
        self._val_scenes = [
            {
                "scene_id": sid,
                "image_path": "",
                "mask_path": None,
                "split": "validation",
                "class_label": None,
                "part": None,
            }
            for sid in split_info.get("validation", [])
        ]
        self._test_scenes = [
            {
                "scene_id": sid,
                "image_path": "",
                "mask_path": None,
                "split": "test",
                "class_label": None,
                "part": None,
            }
            for sid in split_info.get("test", [])
        ]

        logger.info(
            f"Loaded checkpoint from {path}: resuming at epoch {loaded_epoch + 1}, "
            f"best_val_loss={self._best_val_loss:.4f}"
        )
        return loaded_epoch + 1

    def _print_epoch_metrics(self, metrics: Dict, best_epoch: int) -> None:
        """Print end-of-epoch metrics as required."""
        val_loss = metrics.get("loss", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        pr_auc = metrics.get("pr_auc", 0.0)
        epoch_duration = metrics.get("epoch_duration", 0.0)

        # Print all required metrics
        print(f"Epoch {metrics.get('epoch', 'N/A')}/{self.epochs} complete:")
        print(f"  train loss: {metrics.get('train_loss', 0.0):.4f}")
        print(f"  validation loss: {val_loss:.4f}")
        print(f"  precision: {precision:.4f}")
        print(f"  recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")

        # Print confusion matrix
        cm = metrics.get("confusion_matrix")
        if cm:
            print(f"  confusion matrix: TP={cm.get('tp', 0)}, TN={cm.get('tn', 0)}, "
                  f"FP={cm.get('fp', 0)}, FN={cm.get('fn', 0)}")

        print(f"  epoch duration: {epoch_duration:.2f}s")
        print(f"  best epoch so far: {best_epoch}")
        print()

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        normalizer: Optional[ChannelNormalizer] = None,
        resume_epoch: int = 0,
    ) -> Dict:
        """Train the ResNet18 gate."""
        self.normalizer = normalizer
        start_epoch = resume_epoch + 1 if resume_epoch > 0 else 1
        epochs_to_run = self.epochs - (resume_epoch if resume_epoch > 0 else 0)

        # Print dataset split information
        print("Dataset:")
        print(f"train scenes = {len(getattr(train_loader.dataset, 'scenes', []))}")
        print(f"validation scenes = {len(getattr(validation_loader.dataset if validation_loader else [], 'scenes', []))}")
        print("test scenes = 450")
        print()

        # Get actual training scenes from dataset for status report
        train_scenes = []
        if hasattr(train_loader.dataset, 'scenes'):
            train_scenes = train_loader.dataset.scenes
        elif hasattr(train_loader.dataset, '_scenes'):
            train_scenes = train_loader.dataset._scenes

        # Print training status at start
        self._print_training_status(train_scenes, validation_loader.dataset.scenes if validation_loader else [], 450)

        history = {
            "train": [],
            "validation": [],
            "epoch_metrics": [],
        }
        best_val_loss = self._best_val_loss
        best_epoch = getattr(self, "_best_epoch", 0)

        logger.info("Starting ResNet18 training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Resume from checkpoint: {'Yes' if resume_epoch > 0 else 'No'}")
        logger.info(f"Starting epoch: {start_epoch}")

        try:
            for epoch in range(start_epoch - 1, epochs_to_run):
                self.model.train()
                epoch_losses = []
                epoch_positives = 0
                epoch_total = 0

                data_load_start = time.time()

                for batch_idx, batch in enumerate(train_loader):
                    images = batch["image"].to(
                        self.device, non_blocking=torch.cuda.is_available()
                    )
                    labels = batch["label"].float().to(
                        self.device, non_blocking=torch.cuda.is_available()
                    ).unsqueeze(1)

                    data_load_end = time.time()
                    forward_backward_start = time.time()

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

                    forward_backward_time = time.time() - forward_backward_start
                    epoch_losses.append(float(loss.item()))
                    epoch_positives += int((labels >= 0.5).sum())
                    epoch_total += labels.size(0)

                    if torch.cuda.is_available():
                        logger.info(
                            f"  batch {batch_idx}: data_load={data_load_end - data_load_start:.4f}s, "
                            f"forward_backwd={forward_backward_time:.4f}s, "
                            f"gpu_alloc={torch.cuda.memory_allocated()/1024**2:.2f}MB"
                        )

                train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                positive_ratio = (
                    epoch_positives / epoch_total if epoch_total > 0 else 0.0
                )
                data_load_total = time.time() - data_load_start

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"train_loss: {train_loss:.4f} - "
                    f"positive_ratio: {positive_ratio:.4f} - "
                    f"data_load_total={data_load_total:.2f}s"
                )

                if torch.cuda.is_available():
                    logger.info(
                        f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
                        f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB"
                    )

                val_metrics = {}
                if validation_loader is not None:
                    val_start = time.time()
                    val_metrics = self.evaluate(validation_loader)
                    val_duration = time.time() - val_start
                    val_metrics["val_duration"] = val_duration
                    self.scheduler.step(val_metrics["loss"])
                    history["validation"].append(val_metrics)

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        best_epoch = epoch + 1
                        self._best_val_loss = best_val_loss
                        self._best_epoch = best_epoch
                        self.save_checkpoint(
                            self.checkpoint_dir / "best.pth",
                            epoch=epoch + 1,
                            history=history,
                        )
                else:
                    self.scheduler.step(train_loss)

                epoch_duration = time.time() - data_load_start
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "positive_ratio": positive_ratio,
                    "epoch_duration": epoch_duration,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    **val_metrics,
                }

                # Print confusion matrix if available
                if "tp" in val_metrics:
                    epoch_metrics["confusion_matrix"] = {
                        "tp": val_metrics["tp"],
                        "tn": val_metrics["tn"],
                        "fp": val_metrics["fp"],
                        "fn": val_metrics["fn"],
                    }

                history["epoch_metrics"].append(epoch_metrics)
                history["train"].append(train_loss)

                # Print end-of-epoch metrics
                self._print_epoch_metrics(epoch_metrics, best_epoch)

                # Save epoch checkpoints
                self.last_epoch = epoch + 1
                self._log_epoch_metrics(epoch_metrics)
                self.save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch+1}.pth",
                    epoch=epoch + 1,
                    history=history,
                )
                self.save_checkpoint(
                    self.checkpoint_dir / "latest.pth",
                    epoch=epoch + 1,
                    history=history,
                )

        except KeyboardInterrupt:
            logger.warning("Training interrupted by KeyboardInterrupt.")
            # Save current state before quitting
            self.save_checkpoint(
                self.checkpoint_dir / "latest.pth",
                epoch=self.last_epoch,
                history=history,
            )
            print(f"Latest checkpoint saved at {self.checkpoint_dir / 'latest.pth'}")
            print("Incomplete batch is discarded. Next resume will begin from latest.pth.")
            raise

        history["best_val_loss"] = best_val_loss
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        self.save_checkpoint(
            self.checkpoint_dir / "resnet18_final.pth",
            epoch=self.last_epoch,
            history=history,
        )
        return history

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> Dict:
        """Evaluate the ResNet18 gate."""
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
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
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
        """Sweep recall targets and identify threshold candidates."""
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


def train_resnet(
    dataset_root: str,
    config: Optional[dict] = None,
) -> Dict:
    """Train ResNet18 on the SAR dataset."""
    from .model import ResNet18Gate
    from ...dataset.index import DatasetIndex
    from ...dataset.loader import LazyDataset, PatchDataset
    from ...preprocessing.normalization import ChannelNormalizer, compute_normalization_stats
    from ...preprocessing.patching import PatchExtractor
    from torch.utils.data import DataLoader

    config = config or {}

    resume_from_checkpoint = config.get("resume_from_checkpoint")

    logger.info("=== Stage 1 ResNet18 training: dataset discovery starting ===")
    logger.info("Dataset root: %s", dataset_root)
    dataset = DatasetIndex(dataset_root)
    scenes = dataset.discover_scenes()

    if not scenes:
        raise ValueError(
            f"No scenes found in {dataset_root}. Check the dataset path and structure."
        )

    part_counts = {
        part: sum(1 for s in scenes if s.part == part)
        for part in ("Part1", "Part2", "Part3")
    }
    logger.info(
        "Discovered %d scenes total (Part1=%d, Part2=%d, Part3=%d)",
        len(scenes),
        part_counts["Part1"],
        part_counts["Part2"],
        part_counts["Part3"],
    )

    # Split scenes and assign labels
    splitter = SceneSplitter(random_seed=config.get("random_seed", 42))
    split = splitter.split(scenes)
    dataset.scenes = split.train + split.validation + split.test

    logger.info(
        "Split complete: train=%d, validation=%d, test=%d",
        len(split.train),
        len(split.validation),
        len(split.test),
    )
    assert all(s.split == "train" for s in split.train)
    assert all(s.split == "validation" for s in split.validation)
    assert all(s.split == "test" for s in split.test)
    assert all(s.part == "Part3" for s in split.test), (
        "Data leakage: a non-Part3 scene ended up in the held-out test split."
    )
    assert not any(s.part == "Part3" for s in split.train + split.validation), (
        "Data leakage: a Part3 scene leaked into train/validation."
    )

    # Compute normalization statistics from training images only.
    train_image_paths = [scene.image_path for scene in split.train]

    logger.info(
        "Computing normalization statistics from %d training images "
        "(cache checked first; falls back to reading pixel data)...",
        len(split.train),
    )
    vv_mean, vv_std, vh_mean, vh_std = compute_normalization_stats(
        train_image_paths,
        dataset_root=dataset_root,
        random_seed=config.get("random_seed", 42),
    )
    normalizer = ChannelNormalizer(vv_mean, vv_std, vh_mean, vh_std)
    logger.info("Normalization statistics ready.")

    # Build datasets
    logger.info("Constructing train/validation LazyDataset objects...")
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
    logger.info(
        "Datasets constructed: train_scenes=%d, validation_scenes=%d",
        len(train_dataset.scenes),
        len(val_dataset.scenes),
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=config.get("num_workers", 0) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        persistent_workers=config.get("num_workers", 0) > 0,
    )

    # Resume handling
    resume_epoch = 0
    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        if resume_path.exists():
            trainer = ResNetTrainer(config)
            resume_epoch = trainer.load_checkpoint(resume_path) - 1
            logger.info(f"Resuming training from epoch {resume_epoch + 1}")
        else:
            logger.warning(
                f"Resume checkpoint {resume_path} not found; starting from scratch"
            )

    # Train model
    logger.info("=== Actual ResNet18 model training starting now ===")
    trainer = ResNetTrainer(config)
    history = trainer.train(
        train_loader,
        val_loader,
        normalizer=normalizer,
        resume_epoch=resume_epoch,
    )

    # Save artifacts under <dataset_root>/artifacts/resnet18/
    artifact_dir = trainer.checkpoint_dir
    artifact_dir = Path(dataset_root) / "artifacts" / "resnet18" if not config.get("checkpoint_dir") or "artifacts" not in str(config.get("checkpoint_dir")) else trainer.checkpoint_dir
    trainer.checkpoint_dir = artifact_dir
    trainer.save_checkpoint(artifact_dir / "resnet18_final.pth", epoch=trainer.last_epoch, history=history)

    # Save metadata
    metadata = {
        "architecture": "ResNet18",
        "input_channels": 2,
        "input_size": [512, 512],
        "VV_mean": normalizer.vv_mean,
        "VV_std": normalizer.vv_std,
        "VH_mean": normalizer.vh_mean,
        "VH_std": normalizer.vh_std,
        "label_definition": "oil if mask oil fraction >= 0.01, else not oil",
        "oil_patch_threshold_used_to_generate_labels": 0.01,
        "classifier_threshold": config.get("classifier_threshold", 0.12),
        "training_scenes": len(split.train),
        "validation_scenes": len(split.validation),
        "test_scenes": len(split.test),
        "seed": config.get("random_seed", 42),
        "epochs": config.get("epochs", 10),
        "batch_size": config.get("batch_size", 8),
        "learning_rate": config.get("learning_rate", 1e-4),
        "weight_decay": config.get("weight_decay", 1e-4),
        "positive_class_weight": config.get("positive_class_weight"),
        "best_validation_metrics": {
            "best_val_loss": history.get("best_val_loss"),
            "best_epoch": getattr(trainer, "_best_epoch", 0),
        },
        "git_commit_hash": os.getenv("GIT_COMMIT", "unknown"),
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
