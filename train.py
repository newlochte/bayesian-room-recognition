"""
Training entry point for BORM room recognition.

Pipeline:
  1. Training: run YOLOv8 on the train/ images and COUNT, per room,
     in how many images each object appears. Those counts, normalized
     with Laplace smoothing, ARE the trained model:
        P(object | room)  and  P(room).
     There is no gradient descent here — "training" a discrete Bayesian
     network is literally counting.
  2. Evaluation: run the full detection->Bayes pipeline on the held-out
     val/ set, compute metrics, save debug plots.

Note: there is no separate validation set. "Training" is pure counting —
there is no iterative optimization, no early stopping, and no automated
hyperparameter tuning that would consume validation metrics. A validation
set would therefore just be a second, redundant estimate of the same
generalization error, so we evaluate once on the full val/ set. (If you
ever start tuning `smoothing`/`confidence_threshold` automatically, bring
back a validation split so the test estimate stays unbiased.)

Usage:
    python train.py                       # uses ./config.yaml
    python train.py --config other.yaml
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

from utilities.object_detection import ObjectDetector
from utilities.bayesian_inference import BayesianNetwork
from utilities.datasets import build_dataset
from utilities.evaluation import (
    compute_confusion_matrix, plot_confusion_matrix, plot_accuracy_bars,
    plot_object_frequencies, evaluate_split, save_metrics, print_summary,
)


def load_config(path):
    if not os.path.exists(path):
        sys.exit(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Category selection
# ----------------------------------------------------------------------

def select_categories(dataset, selected):
    """Use categories from config, or auto-detect all of them if empty."""
    available = dataset.categories()
    if not selected:
        print(f"No categories selected in config -> using all {len(available)}.")
        return available
    missing = [c for c in selected if c not in available]
    if missing:
        sys.exit(f"Categories not found in dataset: {missing}\n"
                 f"Available examples: {available[:20]} ...")
    return list(selected)


# ----------------------------------------------------------------------
# Detection helpers
# ----------------------------------------------------------------------

def detect_folder(detector, image_paths, batch_size, label=""):
    """Run YOLO over a list of images; returns one object-name set per image."""
    results = []
    start = time.time()
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        for detections in detector.detect_batch(batch):
            results.append(set(detector.object_names(detections)))
        done = min(i + batch_size, len(image_paths))
        rate = done / max(time.time() - start, 1e-6)
        print(f"\r  {label}: {done}/{len(image_paths)} images "
              f"({rate:.1f} img/s)", end="", flush=True)
    print()
    return results


# ----------------------------------------------------------------------
# Main phases
# ----------------------------------------------------------------------

def train_phase(config, dataset, detector, categories):
    """Count object occurrences on train/ and build the Bayesian network."""
    max_per_cat = config["dataset"].get("max_train_images_per_category")
    batch_size = config["detection"].get("batch_size", 16)

    object_names = detector.class_names
    obj_index = {name: j for j, name in enumerate(object_names)}

    # counts[i, j] = in how many train images of room i object j appeared
    counts = np.zeros((len(categories), len(object_names)))
    images_per_room = np.zeros(len(categories))

    print("\n--- Training phase: counting object occurrences per room ---")
    for i, cat in enumerate(categories):
        images = dataset.images(cat, "train")
        if max_per_cat:
            images = images[:max_per_cat]
        images_per_room[i] = len(images)
        per_image_objects = detect_folder(
            detector, images, batch_size, label=f"train/{cat}")
        for objects in per_image_objects:
            for obj in objects:
                counts[i, obj_index[obj]] += 1

    network = BayesianNetwork(categories, object_names)
    network.fit(counts, images_per_room,
                smoothing=config["training"].get("smoothing", 1.0))
    return network


def evaluate_phase(config, dataset, detector, network, split_name):
    """Run detection + Bayes on one split; return posteriors and labels."""
    batch_size = config["detection"].get("batch_size", 16)

    posteriors, labels = [], []
    print(f"\n--- Evaluating on {split_name}/ ---")
    for i, cat in enumerate(network.room_names):
        images = dataset.images(cat, split_name)
        per_image_objects = detect_folder(
            detector, images, batch_size, label=f"{split_name}/{cat}")
        for objects in per_image_objects:
            posteriors.append(network.compute_posterior(sorted(objects)))
            labels.append(i)
    return np.array(posteriors), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Train the BORM Bayesian model")
    parser.add_argument("--config", default="config.yaml",
                        help="path to YAML config (default: ./config.yaml)")
    args = parser.parse_args()
    config = load_config(args.config)

    dataset = build_dataset(config)

    categories = select_categories(
        dataset, config["dataset"].get("selected_categories", []))

    print(f"\nRoom categories ({len(categories)}): {categories}")
    for cat in categories:
        n = len(dataset.images(cat, "train"))
        print(f"  {cat:<18} train={n}")

    detector = ObjectDetector(
        model_name=config["detection"].get("model", "yolov8n"),
        confidence_threshold=config["detection"].get("confidence_threshold", 0.4),
        device=config["detection"].get("device"),
        weights_dir=config["detection"].get("weights_dir", "./models/yolo"),
    )

    # --- Train ---
    network = train_phase(config, dataset, detector, categories)
    model_dir = config["model"].get("dir", "./models/borm")
    network.save(model_dir)
    network.export_structure(
        config["model"].get("network_structure",
                            os.path.join(model_dir, "network_structure.json")))
    print(f"\nTrained model saved to {model_dir}/")

    # --- Evaluate on the held-out val/ set ---
    out_dir = config["training"].get("output_dir", "./output/evaluation")
    os.makedirs(out_dir, exist_ok=True)
    top_k = config["training"].get("top_k_accuracy", 3)

    posteriors, labels = evaluate_phase(config, dataset, detector, network, "val")
    metrics = evaluate_split(posteriors, labels, categories, top_k=top_k)
    print_summary(metrics, "val")

    if config["training"].get("save_plots", True):
        predictions = posteriors.argmax(axis=1)
        cm = compute_confusion_matrix(predictions, labels, len(categories))
        plot_confusion_matrix(
            cm, categories, os.path.join(out_dir, "confusion_matrix.png"))
        plot_accuracy_bars(
            metrics["per_room"], categories,
            os.path.join(out_dir, "accuracy_per_room.png"))
        plot_object_frequencies(
            network, os.path.join(out_dir, "object_frequencies.png"))
        print(f"\nDebug plots saved to {out_dir}/")

    save_metrics({"val": metrics},
                 config["training"].get("metrics_file",
                                        os.path.join(out_dir, "metrics.json")))
    print(f"Metrics saved to {config['training'].get('metrics_file')}")


if __name__ == "__main__":
    main()
