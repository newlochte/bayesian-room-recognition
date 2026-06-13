"""
Inference entry point for BORM room recognition.

Given one image:
  1. detect objects with pre-trained YOLOv8,
  2. compute P(room | objects) with the trained Bayesian network,
  3. print the top-k rooms and save visualizations + results.json.

Usage:
    python infer.py --image path/to/photo.jpg
    python infer.py --config other.yaml --image photo.jpg
"""

import argparse
import json
import os
import sys

import yaml

from utilities.object_detection import ObjectDetector
from utilities.bayesian_inference import BayesianNetwork
from utilities.bayesian_visualization import visualize_network, visualize_inference


def load_config(path):
    if not os.path.exists(path):
        sys.exit(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Predict room type from a photo")
    parser.add_argument("--config", default="config.yaml",
                        help="path to YAML config (default: ./config.yaml)")
    parser.add_argument("--image", default=None,
                        help="path to the input image (overrides config)")
    args = parser.parse_args()
    config = load_config(args.config)

    image_path = args.image or config.get("input", {}).get("image")
    if not image_path or not os.path.exists(image_path):
        sys.exit(f"Input image not found: {image_path}\n"
                 f"Pass one with: python infer.py --image photo.jpg")

    # --- Load trained model ------------------------------------------
    model_dir = config["model"].get("dir", "./models/borm")
    network = BayesianNetwork.load(model_dir)

    # --- Detect objects ----------------------------------------------
    detector = ObjectDetector(
        model_name=config["detection"].get("model", "yolov8n"),
        confidence_threshold=config["detection"].get("confidence_threshold", 0.4),
        device=config["detection"].get("device"),
        weights_dir=config["detection"].get("weights_dir", "./models/yolo"),
    )
    detections = detector.detect(image_path)
    detected_objects = detector.object_names(detections)
    print(f"Detected objects: {detected_objects or '(none)'}")
    if not detected_objects:
        print("No objects above the confidence threshold — the prediction "
              "will fall back to the room priors P(room).")

    # --- Bayesian inference ------------------------------------------
    posterior = network.compute_posterior(detected_objects)
    top_k = config.get("inference", {}).get("top_k", 3)
    top = network.get_top_predictions(detected_objects, k=top_k)

    print(f"\nP(room | objects) — top {top_k}:")
    for rank, (room, prob) in enumerate(top, start=1):
        print(f"  {rank}. {room:<18} {prob:.4f}")
    print(f"\nPredicted room: {top[0][0].upper()}")

    # --- Visualizations ----------------------------------------------
    vis = config.get("visualization", {})
    dpi = vis.get("dpi", 150)
    if vis.get("save_network", True):
        network_path = vis.get("network_path", "./output/inference/bayes_network.png")
        visualize_network(
            network, network_path,
            detected_objects=detected_objects,
            highlight_room=top[0][0],
            max_objects=vis.get("max_objects_in_network", 12),
            dpi=dpi)
        print(f"Network visualization saved to {network_path}")
    if vis.get("save_inference", True):
        inference_path = vis.get("inference_path",
                                 "./output/inference/inference_result.png")
        visualize_inference(
            image_path, detections, posterior, network.room_names,
            inference_path, top_k=top_k, dpi=dpi)
        print(f"Inference visualization saved to {inference_path}")

    # --- Results JSON --------------------------------------------------
    results_path = config.get("output", {}).get(
        "results_json", "./output/inference/results.json")
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    results = {
        "image": str(image_path),
        "detections": detections,
        "detected_objects": detected_objects,
        "posterior": {room: float(p)
                      for room, p in zip(network.room_names, posterior)},
        "top_predictions": [{"room": r, "probability": p} for r, p in top],
        "predicted_room": top[0][0],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
