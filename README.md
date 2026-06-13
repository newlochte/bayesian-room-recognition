# BORM: Bayesian Object Relation Model for Room Recognition

Predict indoor room type from photos using object detection and Bayesian
semantic inference.

This project is a reimplementation based on the BORM paper by Zhou et al.
(2021) and their reference code [[1]](#references) — see `artricle.pdf` for
details. Scene classification relies on the Places365 dataset from MIT CSAIL
[[2]](#references).

The pipeline: **Image → YOLOv8 object detection → Bayesian inference → Room prediction**

```
P(room | objects) ∝ P(room) · Π P(objectᵢ | room)
```

The Bayesian semantic layer (all probability tables) is trained **from
scratch** on Places365 by counting object occurrences per room. Object
detection uses pre-trained YOLOv8 (standard practice — the semantic layer
is the contribution here).

## Installation

```bash
pip install -r requirements.txt
```

YOLOv8 weights download automatically on first use (~6–50 MB depending on
model size).

**Places365 dataset** (manual step — not downloaded automatically):
- Source: http://places.csail.mit.edu/download.html or Kaggle
- Extract to `./data/places365/` so that `train/` and `val/` subfolders
  exist, each with one folder per category.

## Quick Start

**Training:**
```bash
python train.py
```

The model trains on `train/` and is evaluated on the held-out `val/` set.
There's no separate validation split: "training" is pure counting (no
iterative optimization or early stopping), so a validation set would only
be a redundant second estimate of the same generalization error.

**Inference:**
```bash
python infer.py --image path/to/photo.jpg
```

## Configuration

Edit `config.yaml` to customize everything without touching code:

- `dataset.path` — where Places365 lives
- `dataset.selected_categories` — which room categories to use
  (empty list `[]` = all 365). Note: Places365 has no `hallway`/`entrance`
  categories; use `corridor` / `entrance_hall`.
- `dataset.max_train_images_per_category` — main training-time knob
  (object detection is the slow part); `null` = use all images
- `detection.model` — `yolov8n` / `yolov8s` / `yolov8m`
- `detection.confidence_threshold` — detections below this are ignored
- `inference.top_k` — how many room predictions to show

Override the config path with `--config`:
```bash
python train.py --config custom_config.yaml
```

## Acknowledgments

This work is a reimplementation of the **BORM** model. We gratefully acknowledge
the original authors, whose paper and publicly released code [1] form the basis
of this project, as well as the **Places365** dataset [2], provided by MIT CSAIL,
which we use for training and evaluation. All credit for the original method and
dataset belongs to their respective authors; any errors in this reimplementation
are our own.

## References

[1] L. Zhou, J. Cen, X. Wang, Z. Sun, T. L. Lam, and Y. Xu,
"BORM: Bayesian Object Relation Model for Indoor Scene Recognition,"
in *Proc. IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS)*, 2021. arXiv:2108.00397.
Code: https://github.com/FreeformRobotics/BORM

[2] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba,
"Places: A 10 Million Image Database for Scene Recognition,"
*IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*,
vol. 40, no. 6, pp. 1452–1464, 2018. http://places.csail.mit.edu/
