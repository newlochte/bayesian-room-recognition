"""Utility package for the BORM room recognition project."""

from utilities.object_detection import ObjectDetector
from utilities.bayesian_inference import BayesianNetwork
from utilities.bayesian_visualization import visualize_network, visualize_inference
from utilities.evaluation import (
    compute_accuracy,
    compute_topk_accuracy,
    compute_per_class_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_accuracy_bars,
    plot_object_frequencies,
    evaluate_split,
)
