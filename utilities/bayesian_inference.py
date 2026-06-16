"""
Bayesian inference layer for BORM — the core probabilistic graphical model.

The network structure is a simple "naive Bayes" star:

                 Room
               /  |   \
           Obj1  Obj2  Obj3 ...

The Room variable is the (hidden) cause, each Object variable is an
observed effect. Each object variable is binary: "object X appears in
the image" yes/no. Given the room, objects are assumed conditionally
independent — that's the naive Bayes assumption, and it's what lets us
write the joint as a simple product.

Inference is plain Bayes rule with a full Bernoulli likelihood:

    P(room | objects) ∝ P(room) · Π_present P(obj|room) · Π_absent (1 - P(obj|room))

i.e. both the presence AND the absence of each known object count as
evidence (see compute_posterior for why the absence terms matter).
"""

import json
import os

import numpy as np


class BayesianNetwork:
    """Naive-Bayes-style network: Room -> {Object_1, ..., Object_N}."""

    def __init__(self, room_names, object_names):
        self.room_names = list(room_names)
        self.object_names = list(object_names)
        # P_object_given_room[i, j] = P(object j present | room i)
        self.P_object_given_room = None  # shape (num_rooms, num_objects)
        # P_room[i] = prior P(room i)
        self.P_room = None               # shape (num_rooms,)
        # Raw training counts (set by fit(), optionally loaded from disk)
        self.training_counts = None
        self.training_images_per_room = None
        self.training_smoothing = None

    # ------------------------------------------------------------------
    # Training ("from scratch": just counting, the Bayesian way)
    # ------------------------------------------------------------------

    def fit(self, object_presence_counts, images_per_room, smoothing=1.0):
        """
        Estimate the probability tables from training counts.

        object_presence_counts[i, j] = number of training images of room i
                                       in which object j was detected
        images_per_room[i]           = number of training images of room i

        P(object j | room i) is estimated as the fraction of room-i images
        containing object j, with Laplace smoothing so that an object never
        seen in some room still gets a small non-zero probability (otherwise
        a single unexpected detection would zero out that room entirely).
        """
        counts = np.asarray(object_presence_counts, dtype=float)
        n_images = np.asarray(images_per_room, dtype=float)

        # Laplace smoothing for a binary event: add `smoothing` pseudo-count
        # to both the "present" and "absent" outcome.
        self.P_object_given_room = (
            (counts + smoothing) / (n_images[:, None] + 2.0 * smoothing)
        )

        # Room prior = how often each room appears in the training set.
        self.P_room = n_images / n_images.sum()

        # Keep raw counts so they can be saved and used by the animation.
        self.training_counts = counts.astype(int)
        self.training_images_per_room = n_images.astype(int)
        self.training_smoothing = smoothing

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_posterior(self, detected_objects):
        """
        Apply Bayes rule with the FULL Bernoulli naive Bayes likelihood:

            P(room | objects) ∝ P(room)
                                · Π_{obj present} P(obj | room)
                                · Π_{obj absent}  (1 - P(obj | room))

        Every known object is a binary "present / absent" event, so both
        its presence AND its absence are evidence. The absence terms matter
        a lot in practice: a near-empty scene (no bed, no oven, no couch...)
        is itself strong evidence for "empty" rooms like corridors and
        staircases. Without the absence product they could almost never win,
        because they have very few objects to detect, and a broad-object
        room like "office" would absorb everything. (Laplace smoothing in
        fit() keeps every P strictly inside (0, 1), so both log(P) and
        log(1 - P) are always finite.)

        Objects are assumed conditionally independent given the room (the
        naive Bayes assumption), which is what turns the joint into a
        product. Returns a normalized numpy array of shape (num_rooms,).
        """
        # Build a boolean "present" mask over the trained object vocabulary.
        # Detected classes the model never trained on are simply ignored.
        present = np.zeros(len(self.object_names), dtype=bool)
        for obj in detected_objects:
            if obj in self.object_names:
                present[self.object_names.index(obj)] = True

        # Work in log-space: products of many probabilities underflow to 0.0
        # in plain floating point; sums of logs do not.
        log_posterior = np.log(self.P_room).copy()
        log_posterior += np.log(self.P_object_given_room[:, present]).sum(axis=1)
        log_posterior += np.log(1.0 - self.P_object_given_room[:, ~present]).sum(axis=1)

        # Normalize: subtract the max first for numerical stability, then
        # convert back from log-space and divide by the sum (this implements
        # the "/ P(objects)" part of Bayes rule).
        log_posterior -= log_posterior.max()
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum()

    def get_top_predictions(self, detected_objects, k=3):
        """Return the k most likely rooms as (room_name, probability) pairs."""
        posterior = self.compute_posterior(detected_objects)
        top_indices = np.argsort(posterior)[::-1][:k]
        return [(self.room_names[i], float(posterior[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir):
        """Save probability matrices + name mappings to `model_dir`."""
        os.makedirs(model_dir, exist_ok=True)
        np.save(os.path.join(model_dir, "P_object_given_room.npy"),
                self.P_object_given_room)
        np.save(os.path.join(model_dir, "P_room.npy"), self.P_room)
        if self.training_counts is not None:
            np.save(os.path.join(model_dir, "training_counts.npy"),
                    self.training_counts)
            np.save(os.path.join(model_dir, "training_images_per_room.npy"),
                    self.training_images_per_room)
            with open(os.path.join(model_dir, "training_smoothing.json"), "w") as f:
                json.dump(self.training_smoothing, f)
        with open(os.path.join(model_dir, "object_names.json"), "w") as f:
            json.dump(self.object_names, f, indent=2)
        with open(os.path.join(model_dir, "room_names.json"), "w") as f:
            json.dump(self.room_names, f, indent=2)

    @classmethod
    def load(cls, model_dir):
        """Load a trained network from `model_dir`."""
        for required in ("P_object_given_room.npy", "P_room.npy",
                         "object_names.json", "room_names.json"):
            if not os.path.exists(os.path.join(model_dir, required)):
                raise FileNotFoundError(
                    f"Missing '{required}' in '{model_dir}'. "
                    f"Train the model first with: python train.py"
                )
        with open(os.path.join(model_dir, "object_names.json")) as f:
            object_names = json.load(f)
        with open(os.path.join(model_dir, "room_names.json")) as f:
            room_names = json.load(f)
        network = cls(room_names, object_names)
        network.P_object_given_room = np.load(
            os.path.join(model_dir, "P_object_given_room.npy"))
        network.P_room = np.load(os.path.join(model_dir, "P_room.npy"))
        counts_path = os.path.join(model_dir, "training_counts.npy")
        if os.path.exists(counts_path):
            network.training_counts = np.load(counts_path)
            network.training_images_per_room = np.load(
                os.path.join(model_dir, "training_images_per_room.npy"))
            with open(os.path.join(model_dir, "training_smoothing.json")) as f:
                network.training_smoothing = json.load(f)
        return network

    # ------------------------------------------------------------------
    # Export for Manim / external visualization
    # ------------------------------------------------------------------

    def export_structure(self, path, max_objects=15):
        """
        Save the network structure as JSON so it can be re-drawn elsewhere
        (e.g. animated in Manim). Only the `max_objects` most informative
        objects are exported to keep the graph readable; "informative"
        here = highest P(object|room) for its best room.
        """
        best_prob_per_object = self.P_object_given_room.max(axis=0)
        top_objects = np.argsort(best_prob_per_object)[::-1][:max_objects]

        nodes = [{"id": "room", "label": "Room", "type": "root"}]
        edges = []
        for j in top_objects:
            obj = self.object_names[j]
            nodes.append({"id": f"obj_{obj}", "label": obj, "type": "object"})
            edges.append({
                "source": "room",
                "target": f"obj_{obj}",
                # the edge carries the full conditional table, one entry
                # per room, so Manim can animate P(obj|room) per room
                "probability": float(best_prob_per_object[j]),
                "P_object_given_room": {
                    room: float(self.P_object_given_room[i, j])
                    for i, room in enumerate(self.room_names)
                },
            })

        structure = {
            "nodes": nodes,
            "edges": edges,
            "room_categories": self.room_names,
            "object_categories": self.object_names,
            "P_room": {room: float(p)
                       for room, p in zip(self.room_names, self.P_room)},
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(structure, f, indent=2)
        return structure
