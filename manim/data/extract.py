"""
Run from the project root:
    python manim/data/extract.py
Writes manim/data/animation_data.json with all values the scenes need.
"""
import json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, ROOT)

from utilities.bayesian_inference import BayesianNetwork

MODEL_DIR = os.path.join(ROOT, "models/borm")
METRICS_PATH = os.path.join(ROOT, "output/evaluation/metrics.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "animation_data.json")


def build_confusion_matrix(rooms, per_room):
    """Fallback CM (uniform error spread) used only when evaluation did not
    persist the real confusion matrix. Prefer metrics["val"]["confusion_matrix"]."""
    n = len(rooms)
    supports = np.array([per_room[r]["support"] for r in rooms])
    recalls  = np.array([per_room[r]["recall"]  for r in rooms])

    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        tp = round(float(recalls[i] * supports[i]))
        cm[i, i] = tp
        errors = int(supports[i]) - tp
        if errors <= 0:
            continue
        others = [j for j in range(n) if j != i]
        base, rem = divmod(errors, len(others))
        for k, j in enumerate(others):
            cm[i, j] = base + (1 if k < rem else 0)
    return cm.tolist()


def main():
    network = BayesianNetwork.load(MODEL_DIR)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    with open(os.path.join(MODEL_DIR, "network_structure.json")) as f:
        net_structure = json.load(f)

    rooms   = network.room_names
    objects = network.object_names
    P       = network.P_object_given_room          # (n_rooms, n_objects)

    # ── conditional-probability table for Scene 5 ─────────────────────────
    tbl_rooms   = ["office", "bedroom", "kitchen", "bathroom", "living_room", "corridor"]
    tbl_objects = ["chair", "bed", "oven", "sink", "couch", "keyboard", "toilet", "dining table"]
    ri = {r: rooms.index(r)   for r in tbl_rooms}
    oi = {o: objects.index(o) for o in tbl_objects}

    # Use the real training counts saved by train.py if available; otherwise
    # approximate by inverting the Laplace formula (old models won't have them).
    SMOOTHING = float(network.training_smoothing) \
        if network.training_smoothing is not None else 1.0
    if network.training_counts is not None:
        images_per_room = [
            int(network.training_images_per_room[rooms.index(r)])
            for r in tbl_rooms
        ]
        counts = [
            [int(network.training_counts[rooms.index(r), objects.index(o)])
             for o in tbl_objects]
            for r in tbl_rooms
        ]
    else:
        TRAIN_SIZE = 2080
        images_per_room = [
            max(int(round(float(network.P_room[rooms.index(r)]) * TRAIN_SIZE)), 1)
            for r in tbl_rooms
        ]
        counts = [
            [
                max(int(round(float(P[ri[r], oi[o]]) * (images_per_room[i] + 2 * SMOOTHING)
                              - SMOOTHING)), 0)
                for o in tbl_objects
            ]
            for i, r in enumerate(tbl_rooms)
        ]

    table = {
        "rooms":   tbl_rooms,
        "objects": tbl_objects,
        "values": [
            [round(float(P[ri[r], oi[o]]), 3) for o in tbl_objects]
            for r in tbl_rooms
        ],
        "counts":          counts,
        "images_per_room": images_per_room,
        "smoothing":       SMOOTHING,
    }

    # ── demo posteriors for Scene 7 ────────────────────────────────────────
    demo_specs = {
        "kitchen":  {"detected": ["oven", "microwave", "potted plant"], "label": "kuchnia"},
        "bedroom":  {"detected": ["bed", "potted plant", "chair"],      "label": "sypialnia"},
        "corridor": {"detected": [],                                     "label": "korytarz"},
        "office":   {"detected": ["chair", "tv", "keyboard", "book"],   "label": "biuro"},
    }
    for key, spec in demo_specs.items():
        preds = network.get_top_predictions(spec["detected"], k=len(rooms))
        spec["posteriors"] = {r: round(p, 4) for r, p in preds}

    # ── top objects per room (for graph + scene 4) ─────────────────────────
    top_objects_per_room = {}
    for i, room in enumerate(rooms):
        top4 = np.argsort(P[i])[::-1][:4]
        top_objects_per_room[room] = [(objects[j], round(float(P[i, j]), 2)) for j in top4]

    # ── most discriminative objects for the star-graph ────────────────────
    best_prob = P.max(axis=0)
    top12     = np.argsort(best_prob)[::-1][:12]
    graph_objects = [objects[j] for j in top12]

    # ── confusion matrix (real, from evaluation; fallback to approximation) ─
    per_room = metrics["val"]["per_room"]
    cm = metrics["val"].get("confusion_matrix")
    if cm is None:
        cm = build_confusion_matrix(rooms, per_room)

    data = {
        "rooms":   rooms,
        "objects": objects,
        "P_room":  {r: round(float(p), 3) for r, p in zip(rooms, network.P_room)},
        "P_object_given_room": {
            rooms[i]: {objects[j]: round(float(P[i, j]), 3) for j in range(len(objects))}
            for i in range(len(rooms))
        },
        "table":   table,
        "demos":   demo_specs,
        "metrics": {
            "accuracy":      round(metrics["val"]["accuracy"], 3),
            "top3_accuracy": round(metrics["val"]["top3_accuracy"], 3),
            "per_room": {
                r: {k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in m.items()}
                for r, m in per_room.items()
            },
        },
        "graph_objects":       graph_objects,
        "top_objects_per_room": top_objects_per_room,
        "network_structure":   net_structure,
        "confusion_matrix":    cm,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
