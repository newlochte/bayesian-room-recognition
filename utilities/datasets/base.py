"""
Light dataset interface for the BORM pipeline.

Every dataset, whatever its on-disk layout, is exposed through the same two
methods so that train.py / inference never has to know the difference:

    categories()              -> list of room-category names available
    images(category, split)   -> list of image file paths for that split

`split` is one of `SPLITS` ("train", "val"). Datasets that don't ship an
official split synthesize a deterministic one with `deterministic_split`,
so repeated runs always see the same train/val partition.
"""

import os
import random
from abc import ABC, abstractmethod

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_image_files(folder):
    """Sorted image paths directly inside `folder` (empty if it doesn't exist)."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def deterministic_split(items, val_fraction, seed=42):
    """Partition `items` into (train, val) with a stable, seeded shuffle.

    Sorting first makes the result independent of filesystem ordering, so the
    same item always lands in the same split across runs and machines.
    """
    items = sorted(items)
    random.Random(seed).shuffle(items)
    if not items:
        return [], []
    n_val = max(1, round(len(items) * val_fraction))
    return items[n_val:], items[:n_val]


class RoomDataset(ABC):
    """Common contract consumed by training and evaluation."""

    SPLITS = ("train", "val")

    @abstractmethod
    def categories(self):
        """Return the sorted list of available room-category names."""

    @abstractmethod
    def images(self, category, split):
        """Return image paths for one category within one split."""

    def counts(self):
        """{category: {split: n}} — handy for logging/sanity checks."""
        return {
            c: {s: len(self.images(c, s)) for s in self.SPLITS}
            for c in self.categories()
        }
