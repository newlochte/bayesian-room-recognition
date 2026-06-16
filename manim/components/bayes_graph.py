import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
import theme


class BayesStarGraph(VGroup):
    """
    Star-shaped Naive Bayes graph:  Room (center) → Object_i (leaves).
    """

    def __init__(self, object_labels, room_label="Pomieszczenie",
                 radius=2.6, center_r=0.55, leaf_r=0.38,
                 center_color=None, leaf_color=None,
                 font_size_center=22, font_size_leaf=15, **kwargs):
        super().__init__(**kwargs)
        if center_color is None:
            center_color = theme.BAYES_CENTER
        if leaf_color is None:
            leaf_color = theme.BAYES_LEAF

        n = len(object_labels)

        # ── center node ────────────────────────────────────────────────────
        center_circle = Circle(radius=center_r, color=center_color,
                               fill_color=center_color, fill_opacity=0.25,
                               stroke_width=2.5)
        center_text = theme.crisp_text(room_label, font_size=font_size_center,
                           color=center_color).move_to(center_circle)
        self.root = VGroup(center_circle, center_text)   # not 'center' — that shadows VMobject.center()

        # ── leaf nodes + edges ─────────────────────────────────────────────
        self.leaves = VGroup()
        self.edges  = VGroup()
        self.leaf_circles = []

        for i, label in enumerate(object_labels):
            angle = 2 * PI * i / n - PI / 2
            pos   = radius * np.array([np.cos(angle), np.sin(angle), 0])

            dot = Circle(radius=leaf_r, color=leaf_color,
                         fill_color=leaf_color, fill_opacity=0.15,
                         stroke_width=2)
            dot.move_to(pos)

            # wrap long labels
            lines = label.split()
            if len(lines) == 1 or len(label) <= 10:
                txt = theme.crisp_text(label, font_size=font_size_leaf)
            else:
                txt = VGroup(
                    theme.crisp_text(lines[0], font_size=font_size_leaf),
                    theme.crisp_text(" ".join(lines[1:]), font_size=font_size_leaf),
                ).arrange(DOWN, buff=0.04)
            txt.move_to(pos)

            direction = pos / np.linalg.norm(pos)
            edge_start = direction * center_r
            edge_end   = pos - direction * leaf_r

            edge = Line(edge_start, edge_end, color=GRAY_C, stroke_width=1.8)

            self.leaf_circles.append(dot)
            self.edges.add(edge)
            self.leaves.add(VGroup(dot, txt))

        self.add(self.edges, self.root, self.leaves)

    def highlight_leaf(self, idx, color, edge_width=3.5):
        leaf = self.leaves[idx]
        edge = self.edges[idx]
        return AnimationGroup(
            leaf[0].animate.set_color(color).set_fill(color, opacity=0.4),
            leaf[1].animate.set_color(color),
            edge.animate.set_color(color).set_stroke(width=edge_width),
        )

    def dim_leaf(self, idx, color=RED, edge_width=1.0):
        leaf = self.leaves[idx]
        edge = self.edges[idx]
        return AnimationGroup(
            leaf[0].animate.set_color(color).set_fill(color, opacity=0.35),
            leaf[1].animate.set_color(color),
            edge.animate.set_color(color).set_stroke(width=edge_width),
        )
