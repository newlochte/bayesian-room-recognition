"""
Visualization of the Bayesian network and of single-image inference.

Everything is plain matplotlib (no graphviz dependency) so the figures
are easy to tweak and render identically everywhere. Output PNGs are
sized/DPI'd for use as presentation or video frames.
"""

import matplotlib
matplotlib.use("Agg")  # render to files, no display needed

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def visualize_network(network, save_path, detected_objects=None,
                      highlight_room=None, max_objects=12, dpi=150):
    """
    Draw the Room -> Objects star network.

    - Room node on top, object nodes in a row below.
    - Each edge is labeled with P(object | room) for `highlight_room`
      (or for the object's most likely room if none is given).
    - If `detected_objects` is given, those nodes are drawn filled
      (observed evidence), the rest stay white (unobserved) — the
      classic PGM convention.
    """
    detected_objects = set(detected_objects or [])

    # Pick which object nodes to show: detected ones first (they are the
    # evidence we conditioned on), then the most informative remaining ones.
    best_prob = network.P_object_given_room.max(axis=0)
    order = np.argsort(best_prob)[::-1]
    chosen = [j for j in range(len(network.object_names))
              if network.object_names[j] in detected_objects]
    for j in order:
        if len(chosen) >= max_objects:
            break
        if j not in chosen:
            chosen.append(j)
    chosen = chosen[:max_objects]

    if highlight_room is not None:
        room_idx = network.room_names.index(highlight_room)
        room_label = f"Room\n({highlight_room})"
    else:
        room_idx = None
        room_label = "Room"

    n = len(chosen)
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * n), 6))
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, 10)
    ax.axis("off")

    room_xy = ((n + 1) / 2, 8.5)

    # Object nodes + edges first (so the room node draws on top of edges)
    for pos, j in enumerate(chosen, start=1):
        obj = network.object_names[j]
        obj_xy = (pos, 2.0)
        observed = obj in detected_objects

        # Edge probability: P(obj | highlighted room), or the object's
        # best room if we are drawing the generic (pre-evidence) network.
        if room_idx is not None:
            p = network.P_object_given_room[room_idx, j]
        else:
            p = best_prob[j]

        ax.annotate(
            "", xy=(obj_xy[0], obj_xy[1] + 0.75), xytext=(room_xy[0], room_xy[1] - 0.9),
            arrowprops=dict(arrowstyle="-|>", lw=1.6 if observed else 1.0,
                            color="#2b6cb0" if observed else "#a0aec0"),
        )
        # Edge label sits at the midpoint of the edge
        mid = ((obj_xy[0] + room_xy[0]) / 2, (obj_xy[1] + room_xy[1]) / 2)
        ax.text(mid[0], mid[1], f"{p:.2f}", fontsize=8, ha="center",
                color="#2b6cb0" if observed else "#718096",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.8))

        circle = mpatches.Ellipse(
            obj_xy, 0.85, 1.5,
            facecolor="#bee3f8" if observed else "white",
            edgecolor="#2b6cb0" if observed else "#4a5568", lw=1.4)
        ax.add_patch(circle)
        ax.text(obj_xy[0], obj_xy[1], obj.replace(" ", "\n"), fontsize=8,
                ha="center", va="center")

    # Room node (the hidden variable we infer)
    room_node = mpatches.Ellipse(room_xy, 1.9, 1.9, facecolor="#fefcbf",
                                 edgecolor="#b7791f", lw=2)
    ax.add_patch(room_node)
    ax.text(room_xy[0], room_xy[1], room_label, fontsize=11,
            ha="center", va="center", weight="bold")

    # Legend explaining the PGM conventions
    legend_items = [
        mpatches.Patch(fc="#fefcbf", ec="#b7791f", label="hidden: Room"),
        mpatches.Patch(fc="#bee3f8", ec="#2b6cb0", label="observed object"),
        mpatches.Patch(fc="white", ec="#4a5568", label="unobserved object"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)
    ax.set_title("BORM Bayesian network:  P(room | objects) ∝ P(room) · Π P(objectᵢ | room)",
                 fontsize=12)

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_inference(image_path, detections, posterior, room_names,
                        save_path, top_k=3, dpi=150):
    """
    Two-panel inference summary:
      left  — the input image with detection bounding boxes,
      right — detected objects, posterior bar chart, and the Bayes
              computation written out for the winning room.
    """
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1])
    ax_img = fig.add_subplot(grid[:, 0])
    ax_bars = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    # --- Left panel: image + boxes -----------------------------------
    ax_img.imshow(image)
    ax_img.axis("off")
    colors = plt.cm.tab10.colors
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        color = colors[i % len(colors)]
        ax_img.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, lw=2, edgecolor=color))
        ax_img.text(x1, max(y1 - 5, 8), f'{det["name"]} {det["confidence"]:.2f}',
                    fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"))
    ax_img.set_title("Input image + YOLOv8 detections", fontsize=12)

    # --- Top-right panel: posterior bar chart ------------------------
    order = np.argsort(posterior)[::-1]
    shown = order[:max(top_k, 5)]
    probs = [posterior[i] for i in shown]
    labels = [room_names[i] for i in shown]
    bar_colors = ["#2f855a" if i == order[0] else "#a0aec0" for i in shown]
    y_pos = np.arange(len(shown))[::-1]
    ax_bars.barh(y_pos, probs, color=bar_colors)
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(labels, fontsize=10)
    ax_bars.set_xlim(0, 1)
    ax_bars.set_xlabel("P(room | objects)")
    ax_bars.set_title("Posterior over rooms", fontsize=12)
    for y, p in zip(y_pos, probs):
        ax_bars.text(min(p + 0.01, 0.93), y, f"{p:.3f}", va="center", fontsize=9)

    # --- Bottom-right panel: the Bayes computation, written out ------
    ax_text.axis("off")
    detected_names = sorted({d["name"] for d in detections})
    winner = room_names[order[0]]
    lines = [f"Detected objects: {', '.join(detected_names) or '(none)'}", ""]
    lines.append("Bayes rule:  P(room | objects) ∝ P(room) · Π P(objᵢ | room)")
    lines.append("")
    for rank, i in enumerate(order[:top_k], start=1):
        lines.append(f"{rank}. {room_names[i]:<16} P = {posterior[i]:.4f}")
    lines.append("")
    lines.append(f"Prediction:  {winner.upper()}")
    ax_text.text(0.02, 0.95, "\n".join(lines), fontsize=11, family="monospace",
                 va="top", transform=ax_text.transAxes)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
