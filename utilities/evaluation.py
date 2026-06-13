"""
Metrics and debug plots for the trained Bayesian model.

All the heavy lifting is done by scikit-learn; this module just wraps it
with our room-name bookkeeping and saves presentation-quality figures.
"""

import matplotlib
matplotlib.use("Agg")

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------

def compute_accuracy(predictions, ground_truth):
    """Fraction of images whose top-1 predicted room is correct."""
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    return float((predictions == ground_truth).mean())


def compute_topk_accuracy(posteriors, ground_truth, k=3):
    """How often the true room is among the k most probable rooms."""
    posteriors = np.asarray(posteriors)          # (n_images, n_rooms)
    ground_truth = np.asarray(ground_truth)      # (n_images,) int labels
    topk = np.argsort(posteriors, axis=1)[:, ::-1][:, :k]
    hits = (topk == ground_truth[:, None]).any(axis=1)
    return float(hits.mean())


def compute_per_class_metrics(predictions, ground_truth, class_names):
    """Precision / recall / F1 for every room, plus macro averages."""
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, zero_division=0)
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    macro = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    return per_class, macro


def compute_confusion_matrix(predictions, ground_truth, num_classes):
    return confusion_matrix(ground_truth, predictions,
                            labels=list(range(num_classes)))


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_confusion_matrix(cm, room_names, save_path, dpi=150):
    """Heatmap of which rooms get confused with which."""
    # Normalize per row so each row reads as P(predicted | true room)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(room_names)))
    ax.set_yticks(range(len(room_names)))
    ax.set_xticklabels(room_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(room_names, fontsize=9)
    ax.set_xlabel("Predicted room")
    ax.set_ylabel("True room")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(room_names)):
        for j in range(len(room_names)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bars(per_class_metrics, room_names, save_path, dpi=150):
    """Per-room precision/recall/F1 as grouped bars."""
    x = np.arange(len(room_names))
    width = 0.27
    precision = [per_class_metrics[r]["precision"] for r in room_names]
    recall = [per_class_metrics[r]["recall"] for r in room_names]
    f1 = [per_class_metrics[r]["f1"] for r in room_names]

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(room_names)), 5))
    ax.bar(x - width, precision, width, label="precision", color="#2b6cb0")
    ax.bar(x, recall, width, label="recall", color="#2f855a")
    ax.bar(x + width, f1, width, label="F1", color="#b7791f")
    ax.set_xticks(x)
    ax.set_xticklabels(room_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Per-room metrics (test set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_object_frequencies(network, save_path, top_n=8, dpi=150):
    """
    For each room, show the top-N objects by P(object | room).
    This is the best sanity check of training: 'bed' should dominate
    bedroom, 'toilet'/'sink' should dominate bathroom, etc.
    """
    rooms = network.room_names
    n_rooms = len(rooms)
    ncols = 3
    nrows = int(np.ceil(n_rooms / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.0 * nrows), squeeze=False)
    for idx, room in enumerate(rooms):
        ax = axes[idx // ncols][idx % ncols]
        probs = network.P_object_given_room[idx]
        top = np.argsort(probs)[::-1][:top_n]
        names = [network.object_names[j] for j in top][::-1]
        values = [probs[j] for j in top][::-1]
        ax.barh(names, values, color="#2b6cb0")
        ax.set_title(room, fontsize=11)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)
    # hide unused subplots
    for idx in range(n_rooms, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Most likely objects per room — P(object | room)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Full evaluation entry point (used by train.py)
# ----------------------------------------------------------------------

def evaluate_split(posteriors, ground_truth, room_names, top_k=3):
    """Compute every metric for one dataset split; returns a dict."""
    posteriors = np.asarray(posteriors)
    predictions = posteriors.argmax(axis=1)
    per_class, macro = compute_per_class_metrics(
        predictions, ground_truth, room_names)
    return {
        "accuracy": compute_accuracy(predictions, ground_truth),
        f"top{top_k}_accuracy": compute_topk_accuracy(
            posteriors, ground_truth, k=top_k),
        "macro": macro,
        "per_room": per_class,
        "num_images": len(ground_truth),
    }


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def print_summary(metrics, split_name):
    print(f"\n=== {split_name} results ({metrics['num_images']} images) ===")
    print(f"accuracy        : {metrics['accuracy']:.3f}")
    topk_key = next(k for k in metrics if k.startswith("top"))
    print(f"{topk_key:<16}: {metrics[topk_key]:.3f}")
    print(f"macro F1        : {metrics['macro']['f1']:.3f}")
    print(f"{'room':<18}{'prec':>7}{'rec':>7}{'F1':>7}{'n':>6}")
    for room, m in metrics["per_room"].items():
        print(f"{room:<18}{m['precision']:>7.2f}{m['recall']:>7.2f}"
              f"{m['f1']:>7.2f}{m['support']:>6}")
