"""MIT Indoor-67: <root>/Images/<category>/*.jpg, with no official split.

Each category is a single image pool; we synthesize a deterministic
train/val split per category so every category is represented in both.
"""

import os

from .base import RoomDataset, deterministic_split, list_image_files


class Indoor67Dataset(RoomDataset):
    def __init__(self, root, val_fraction=0.15, seed=42, **_):
        self.images_root = os.path.join(root, "Images")
        if not os.path.isdir(self.images_root):
            raise SystemExit(f"Expected '{self.images_root}/<category>/*.jpg'")
        self.val_fraction = val_fraction
        self.seed = seed

    def categories(self):
        return sorted(
            d for d in os.listdir(self.images_root)
            if os.path.isdir(os.path.join(self.images_root, d))
        )

    def images(self, category, split):
        pool = list_image_files(os.path.join(self.images_root, category))
        train, val = deterministic_split(pool, self.val_fraction, self.seed)
        return {"train": train, "val": val}[split]
