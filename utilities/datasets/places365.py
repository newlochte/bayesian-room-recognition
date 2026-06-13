"""Places365 layout: <root>/<split>/<category>/*.jpg.

The trivial case: the split and categories are just the folders. Also fits
any dataset materialized into this same layout (e.g. prepare_sunrgbd.py).
"""

import os

from .base import RoomDataset, list_image_files


class Places365Dataset(RoomDataset):
    def __init__(self, root, **_):
        self.root = root
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
            raise SystemExit(
                f"Expected '{root}/train' and '{root}/val', each with one\n"
                f"subfolder per category. Got: {root}"
            )

    def categories(self):
        train_dir = os.path.join(self.root, "train")
        return sorted(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )

    def images(self, category, split):
        return list_image_files(os.path.join(self.root, split, category))
