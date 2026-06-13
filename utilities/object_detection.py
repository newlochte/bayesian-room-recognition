"""
Object detection layer for BORM.

This wraps a pre-trained YOLOv8 model from the `ultralytics` package.
We do NOT train the detector ourselves — BORM's contribution is the
Bayesian semantic layer on top of detections, so using an off-the-shelf
detector is standard practice (the paper does the same).

YOLOv8 is trained on COCO, so it knows 80 everyday object classes such as
"bed", "toilet", "oven", "couch", "tv", "sink", "chair", ... Many of these
are strongly tied to specific rooms, which is exactly what the Bayesian
layer exploits.
"""

import os

from ultralytics import YOLO


class ObjectDetector:
    """Thin wrapper around a pre-trained YOLOv8 model."""

    def __init__(self, model_name="yolov8n", confidence_threshold=0.4,
                 device=None, weights_dir="./models/yolo"):
        # Keep the downloaded YOLO weights in their own folder (models/yolo/)
        # rather than the project root. ultralytics downloads the weights
        # automatically on first use to the path we hand it
        # (e.g. "yolov8n.pt" ~6MB for nano, ~50MB for medium), then reuses
        # the cached file on later runs.
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, model_name + ".pt")
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        # COCO class names, e.g. {0: "person", 1: "bicycle", ...}
        self.class_names = [self.model.names[i] for i in sorted(self.model.names)]

    def detect(self, image_path):
        """
        Detect objects in a single image.

        Returns a list of detections, each a dict:
            {"name": "bed", "confidence": 0.91, "box": [x1, y1, x2, y2]}
        """
        return self.detect_batch([image_path])[0]

    def detect_batch(self, image_paths):
        """
        Detect objects in a batch of images (much faster on GPU than
        one-by-one). Returns one detection list per input image.
        """
        results = self.model.predict(
            [str(p) for p in image_paths],
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        all_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append({
                    "name": result.names[class_id],
                    "confidence": float(box.conf.item()),
                    "box": [float(v) for v in box.xyxy[0].tolist()],
                })
            all_detections.append(detections)
        return all_detections

    @staticmethod
    def object_names(detections):
        """Reduce detections to the set of distinct object names.

        We only care about *which* objects are present, not how many:
        the Bayesian model uses Bernoulli "object present in image"
        events, so duplicates would wrongly multiply the same evidence
        twice.
        """
        return sorted({d["name"] for d in detections})
