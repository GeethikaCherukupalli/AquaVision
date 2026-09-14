# -*- coding: utf-8 -*-
"""Scene-level data splitting for leakage prevention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .index import SceneIndexItem


@dataclass
class SceneSplit:
    """Scene-level train/validation/test split."""

    train: List[SceneIndexItem]
    validation: List[SceneIndexItem]
    test: List[SceneIndexItem]

    @property
    def all_scenes(self) -> List[SceneIndexItem]:
        return self.train + self.validation + self.test


class SceneSplitter:
    """Split Part1/Part2 into train/validation while preserving Part3 as test.

    Part3 is the official held-out test set and is NEVER randomly
    repartitioned into train or validation.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def split(
        self,
        scenes: List[SceneIndexItem],
        train_ratio: float = 0.80,
        validation_ratio: float = 0.20,
        test_ratio: float = 0.0,
        per_class: bool = True,
    ) -> SceneSplit:

        if abs(train_ratio + validation_ratio - 1.0) > 1e-8:
            raise ValueError(
                "Train and validation ratios must sum to 1.0 "
                "when Part3 is the fixed test set."
            )

        rng = np.random.RandomState(self.random_seed)

        # Part3 is already explicitly marked test by DatasetIndex.
        fixed_test = [
            scene for scene in scenes
            if scene.part == "Part3" or scene.split == "test"
        ]

        # Only Part1/Part2 are eligible for train/validation.
        train_val = [
            scene for scene in scenes
            if scene.part != "Part3" and scene.split != "test"
        ]

        if not train_val:
            raise ValueError("No Part1/Part2 scenes available for train/validation.")

        if per_class:
            train, validation = self._split_per_class(
                train_val,
                train_ratio,
                validation_ratio,
                rng,
            )
        else:
            rng.shuffle(train_val)

            n_train = int(len(train_val) * train_ratio)

            train = train_val[:n_train]
            validation = train_val[n_train:]

        rng.shuffle(train)
        rng.shuffle(validation)
        rng.shuffle(fixed_test)

        return SceneSplit(
            train=train,
            validation=validation,
            test=fixed_test,
        )

    def _split_per_class(
        self,
        scenes: List[SceneIndexItem],
        train_ratio: float,
        validation_ratio: float,
        rng: np.random.RandomState,
    ):
        """Stratified scene-level train/validation split."""

        groups = {}

        for scene in scenes:
            label = scene.class_label or "unknown"
            groups.setdefault(label, []).append(scene)

        train = []
        validation = []

        for label, group in groups.items():
            rng.shuffle(group)

            n_train = int(len(group) * train_ratio)

            # Keep at least one validation scene for sufficiently
            # large groups.
            if len(group) > 1:
                n_train = min(n_train, len(group) - 1)

            train.extend(group[:n_train])
            validation.extend(group[n_train:])

        return train, validation