# -*- coding: utf-8 -*-
"""Scene-level data splitting for leakage prevention."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .index import DatasetIndex, SceneIndexItem


@dataclass
class SceneSplit:
    """Scene-level train/validation/test split."""

    train: List[SceneIndexItem]
    validation: List[SceneIndexItem]
    test: List[SceneIndexItem]

    @property
    def all_scenes(self) -> List[SceneIndexItem]:
        """Return all scenes."""
        return self.train + self.validation + self.test


class SceneSplitter:
    """Split scenes into train/validation/test with leakage prevention.

    The split is performed at the scene level, never at the patch level.
    This ensures that spatially adjacent patches from the same scene
    never appear in different splits.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def split(
        self,
        scenes: List[SceneIndexItem],
        train_ratio: float = 0.80,
        validation_ratio: float = 0.10,
        test_ratio: float = 0.10,
        per_class: bool = True,
    ) -> SceneSplit:
        """Split scenes into train/validation/test.

        Parameters
        ----------
        scenes : list of SceneIndexItem
            All discovered scenes.
        train_ratio : float
            Fraction of scenes for training.
        validation_ratio : float
            Fraction of scenes for validation.
        test_ratio : float
            Fraction of scenes for test (held-out Part III).
        per_class : bool
            If True, split stratified by class label.

        Returns
        -------
        SceneSplit
        """
        if train_ratio + validation_ratio + test_ratio != 1.0:
            raise ValueError("Splits must sum to 1.0")

        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(scenes)

        total = len(scenes)
        if total == 0:
            raise ValueError("No scenes available for splitting")

        if per_class and any(scene.class_label for scene in scenes):
            return self._split_per_class(
                scenes, train_ratio, validation_ratio, test_ratio, rng
            )

        # Non-stratified split
        n_train = int(total * train_ratio)
        n_validation = int(total * validation_ratio)
        n_test = total - n_train - n_validation

        train_end = n_train
        validation_end = n_train + n_validation

        return SceneSplit(
            train=scenes[:train_end],
            validation=scenes[train_end:validation_end],
            test=scenes[validation_end:],
        )

    def _split_per_class(
        self,
        scenes: List[SceneIndexItem],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        rng: np.random.RandomState,
    ) -> SceneSplit:
        """Stratified split by class label."""
        groups: dict = {}
        for scene in scenes:
            label = scene.class_label or "unknown"
            groups.setdefault(label, []).append(scene)

        train: List[SceneIndexItem] = []
        validation: List[SceneIndexItem] = []
        test: List[SceneIndexItem] = []

        for label, group in groups.items():
            rng.shuffle(group)
            n_train = max(1, int(len(group) * train_ratio)) if len(group) > 1 else 0
            n_validation = (
                max(1, int(len(group) * validation_ratio)) if len(group) > 1 else 0
            )
            n_test = len(group) - n_train - n_validation

            train.extend(group[:n_train])
            validation.extend(group[n_train : n_train + n_validation])
            test.extend(group[n_train + n_validation :])

        rng.shuffle(train)
        rng.shuffle(validation)
        rng.shuffle(test)
        return SceneSplit(train=train, validation=validation, test=test)