"""Raw SUN RGB-D, read in place (no prepare_sunrgbd.py step needed).

Each capture is a folder containing `scene.txt` (the label) and
`image/<x>.jpg` (the RGB frame). We group captures by scene label and
synthesize a deterministic train/val split per category.
"""

import os

from .base import RoomDataset, deterministic_split

SKIP_LABELS = {"idk", ""}


class SunRGBDDataset(RoomDataset):
    def __init__(self, root, val_fraction=0.15, seed=42,
                 category_map=None, min_images=1, **_):
        self.root = root
        self.val_fraction = val_fraction
        self.seed = seed
        self.category_map = category_map or {}
        self.min_images = min_images
        self._by_label = None  # lazily built {label: [image_path, ...]}

    def _scan(self):
        if self._by_label is not None:
            return self._by_label
        by_label = {}
        for dirpath, _dirs, files in os.walk(self.root):
            if "scene.txt" not in files:
                continue
            with open(os.path.join(dirpath, "scene.txt")) as f:
                label = f.read().strip()
            if label in SKIP_LABELS:
                continue
            label = self.category_map.get(label, label)
            image_dir = os.path.join(dirpath, "image")
            if not os.path.isdir(image_dir):
                continue
            jpgs = sorted(n for n in os.listdir(image_dir)
                          if n.lower().endswith(".jpg"))
            if jpgs:
                by_label.setdefault(label, []).append(
                    os.path.join(image_dir, jpgs[0]))
        self._by_label = {
            lbl: paths for lbl, paths in by_label.items()
            if len(paths) >= self.min_images
        }
        return self._by_label

    def categories(self):
        return sorted(self._scan())

    def images(self, category, split):
        pool = self._scan().get(category, [])
        train, val = deterministic_split(pool, self.val_fraction, self.seed)
        return {"train": train, "val": val}[split]
