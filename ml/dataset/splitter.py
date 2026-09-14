from dataclasses import dataclass
from typing import List

import numpy as np

from .index import SceneIndexItem


@dataclass
class SceneSplit:
    train: List[SceneIndexItem]
    validation: List[SceneIndexItem]
    test: List[SceneIndexItem]

    @property
    def all_scenes(self) -> List[SceneIndexItem]:
        return self.train + self.validation + self.test


class SceneSplitter:
    """
    Scene-level splitter.

    Important:
    - Part3 is permanently held out as TEST.
    - Only Part1 and Part2 are split into TRAIN/VALIDATION.
    - Splitting happens at scene level to prevent patch leakage.
    """

    def __init__(self, random_seed: int = 42):
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
                "train_ratio + validation_ratio must equal 1.0 "
                "because Part3 is the fixed test set."
            )

        if test_ratio != 0.0:
            raise ValueError(
                "test_ratio must be 0.0 because Part3 is already "
                "the fixed held-out test set."
            )

        rng = np.random.RandomState(self.random_seed)

        # ---------------------------------------------------------
        # FIXED TEST SET
        # ---------------------------------------------------------
        fixed_test = [
            scene
            for scene in scenes
            if scene.part == "Part3" or scene.split == "test"
        ]

        # ---------------------------------------------------------
        # TRAIN + VALIDATION CANDIDATES
        # ---------------------------------------------------------
        train_val = [
            scene
            for scene in scenes
            if scene.part != "Part3" and scene.split != "test"
        ]

        if not train_val:
            raise ValueError("No training/validation scenes available.")

        if per_class:
            train, validation = self._split_per_class(
                train_val,
                train_ratio,
                validation_ratio,
                rng,
            )
        else:
            shuffled = list(train_val)
            rng.shuffle(shuffled)

            n_train = int(len(shuffled) * train_ratio)

            train = shuffled[:n_train]
            validation = shuffled[n_train:]

        # ---------------------------------------------------------
        # STAMP scene.split TO MATCH THE PARTITION IT WAS PLACED IN.
        #
        # discover_scenes() marks every Part1/Part2 scene as "train"
        # up front (see dataset/index.py), on the assumption that this
        # splitter would assign the real per-scene split afterwards.
        # Previously that assignment never happened: scenes placed in
        # `validation` kept the literal string "train" on
        # `scene.split`, so LazyDataset(split="validation") filtered
        # dataset.scenes down to an empty list and raised
        # "No scenes found for split 'validation'". Stamping here,
        # at the single source of truth, means every caller of
        # SceneSplitter (train_resnet, tests, notebooks) gets scenes
        # whose `.split` field is always correct without needing an
        # ad-hoc fix downstream.
        # ---------------------------------------------------------
        for scene in train:
            scene.split = "train"
        for scene in validation:
            scene.split = "validation"
        for scene in fixed_test:
            scene.split = "test"

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
        """
        Stratified scene-level train/validation split.
        """

        classes = sorted(set(scene.class_label for scene in scenes))

        train = []
        validation = []

        for class_label in classes:

            class_scenes = [
                scene
                for scene in scenes
                if scene.class_label == class_label
            ]

            rng.shuffle(class_scenes)

            n_train = int(len(class_scenes) * train_ratio)

            # Guarantee at least one validation scene
            # when a class has more than one scene.
            if len(class_scenes) > 1:
                n_train = min(
                    n_train,
                    len(class_scenes) - 1,
                )

            class_train = class_scenes[:n_train]
            class_validation = class_scenes[n_train:]

            train.extend(class_train)
            validation.extend(class_validation)

        rng.shuffle(train)
        rng.shuffle(validation)

        return train, validation