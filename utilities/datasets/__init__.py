"""Dataset adapters for the BORM pipeline.

All datasets are consumed through the `RoomDataset` interface, so training
and evaluation are identical regardless of on-disk layout. Pick an adapter
with `dataset.loader` in config.yaml, or leave it unset to auto-detect from
the directory structure.

    from utilities.datasets import build_dataset
    dataset = build_dataset(config)
    dataset.categories()              # ["bedroom", "kitchen", ...]
    dataset.images("kitchen", "train")  # ["/.../a.jpg", ...]
"""

import os

from .base import RoomDataset, deterministic_split, list_image_files
from .places365 import Places365Dataset
from .indoor67 import Indoor67Dataset
from .sunrgbd import SunRGBDDataset

# Registry of loader name -> adapter class.
REGISTRY = {
    "places365": Places365Dataset,  # <root>/<split>/<category>/*.jpg
    "indoor67": Indoor67Dataset,
    "sunrgbd": SunRGBDDataset,      # raw SUN RGB-D, no prep step
}


def _infer_loader(root):
    """Guess the adapter from the directory structure."""
    if os.path.isdir(os.path.join(root, "train")) and \
            os.path.isdir(os.path.join(root, "val")):
        return "places365"
    if os.path.isdir(os.path.join(root, "Images")):
        return "indoor67"
    # Any capture folder with a scene.txt => raw SUN RGB-D.
    for dirpath, _dirs, files in os.walk(root):
        if "scene.txt" in files:
            return "sunrgbd"
    raise SystemExit(
        f"Could not infer dataset loader for '{root}'. Set `dataset.loader` "
        f"in config to one of: {sorted(REGISTRY)}"
    )


def build_dataset(config):
    """Construct the RoomDataset described by `config['dataset']`."""
    dcfg = config["dataset"]
    root = dcfg["path"]
    if not os.path.isdir(root):
        raise SystemExit(f"Dataset path not found: {root}")

    loader = dcfg.get("loader") or _infer_loader(root)
    try:
        cls = REGISTRY[loader]
    except KeyError:
        raise SystemExit(
            f"Unknown dataset loader '{loader}'. "
            f"Available: {sorted(REGISTRY)}")

    return cls(
        root,
        val_fraction=dcfg.get("val_fraction", 0.15),
        seed=dcfg.get("seed", 42),
        category_map=dcfg.get("category_map"),
        min_images=dcfg.get("min_images", 1),
    )


__all__ = [
    "RoomDataset", "deterministic_split", "list_image_files",
    "Places365Dataset", "Indoor67Dataset", "SunRGBDDataset",
    "REGISTRY", "build_dataset",
]
