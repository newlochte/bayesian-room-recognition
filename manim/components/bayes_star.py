import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
from manim import *
import theme


def naive_bayes_posterior(room_labels, weights_table, present, absent,
                          default=0.05, eps=1e-6):
    """Bernoulli naive-Bayes posterior over rooms.

    Mirrors utilities/bayesian_inference.compute_posterior: a uniform prior
    times P(obj|room) for every PRESENT object and (1 - P(obj|room)) for every
    ABSENT one, in log-space, then normalized. `weights_table[room][obj]` is
    P(obj present | room); unknown pairs fall back to `default`. Objects that are
    neither in `present` nor `absent` are simply unobserved and contribute
    nothing.
    """
    logp = np.zeros(len(room_labels))
    for ri, room in enumerate(room_labels):
        t = weights_table.get(room, {})
        s = 0.0
        for o in present:
            s += math.log(max(eps, t.get(o, default)))
        for o in absent:
            s += math.log(max(eps, 1.0 - t.get(o, default)))
        logp[ri] = s
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


class BayesStarGraph(VGroup):
    """Naive-Bayes *structure*: a single Room node -> many object nodes.

        ┌──────────── Pomieszczenie ────────────┐
        │  ▁  ▂  ▁  █  ▁   ← posterior bars      │
        └──────────────────┬────────────────────┘
              (directed arrows, radial fan)
           ●     ●     ●     ●     ●     ●
         obj_0 obj_1 ...

    The structure is correct (ONE room variable), but the room node carries a
    live bar chart over the room values, so "many inputs shape what's probable"
    stays visible. There are deliberately no edges between objects — that is the
    conditional-independence ("naive") assumption, drawn explicitly by
    `independence_demo()`.
    """

    # Short labels for the in-node bar chart (room labels are wide).
    _ABBR = {"Kuchnia": "Kuch", "Salon": "Salon", "Sypialnia": "Syp",
             "Biuro": "Biuro", "Łazienka": "Łaz"}

    def __init__(self, room_labels, object_labels,
                 node_w=6.4, node_h=2.0, node_y=2.45,
                 object_y=-2.65, object_width=12.0, object_r=0.34,
                 font_size_object=15, room_color=None, object_color=None,
                 arrow_color=None, **kwargs):
        super().__init__(**kwargs)
        self.room_labels = list(room_labels)
        self.object_labels = list(object_labels)
        self.room_color = room_color or theme.BAYES_CENTER
        self.object_color = object_color or theme.BAYES_LEAF
        arrow_color = arrow_color or theme.EDGE
        K = len(self.room_labels)

        # ── room node (single, top) ────────────────────────────────────────
        box = RoundedRectangle(corner_radius=0.14, width=node_w, height=node_h,
                               color=self.room_color, fill_color=self.room_color,
                               fill_opacity=0.10, stroke_width=2.5)
        box.move_to([0, node_y, 0])
        title = theme.crisp_text("Pomieszczenie", font_size=20,
                                 color=self.room_color)
        title.move_to(box.get_top() + DOWN * 0.26)

        # in-node posterior bar chart (starts uniform)
        self.bar_baseline_y = node_y - node_h / 2 + 0.42
        self.bar_max_h = 0.95
        bar_xs = np.linspace(-node_w / 2 + 0.85, node_w / 2 - 0.85, K)
        bar_w = min(0.62, (bar_xs[1] - bar_xs[0]) * 0.6) if K > 1 else 0.6
        self.bar_xs = bar_xs
        self._bar_values = np.full(K, 1.0 / K)

        self.bars = VGroup()
        self.bar_labels = VGroup()
        for x, room in zip(bar_xs, self.room_labels):
            h = max(0.04, self._bar_values[0] * self.bar_max_h)
            bar = Rectangle(width=bar_w, height=h, stroke_width=0,
                            fill_color=self.room_color, fill_opacity=0.55)
            bar.move_to([x, self.bar_baseline_y + h / 2, 0])
            lbl = theme.crisp_text(self._ABBR.get(room, room[:5]),
                                   font_size=12, color=theme.MUTED)
            lbl.move_to([x, self.bar_baseline_y - 0.22, 0])
            self.bars.add(bar)
            self.bar_labels.add(lbl)

        self.room_node = VGroup(box, title, self.bars, self.bar_labels)
        self.room_box = box

        # ── object nodes (binary present/absent) ───────────────────────────
        self.objects = VGroup()
        self.object_circles = []
        xs = self._spread(len(object_labels), object_width)
        for x, label in zip(xs, object_labels):
            dot = Circle(radius=object_r, color=self.object_color,
                         fill_color=self.object_color, fill_opacity=0.15,
                         stroke_width=2)
            dot.move_to([x, object_y, 0])
            txt = theme.crisp_text(label, font_size=font_size_object,
                                   color=self.object_color)
            txt.next_to(dot, DOWN, buff=0.12)
            self.object_circles.append(dot)
            self.objects.add(VGroup(dot, txt))

        # ── directed arrows: radial fan from node bottom to each object ─────
        fan_origin = box.get_bottom()
        self.arrows = []
        self.arrow_paths = []          # invisible Lines used as flow paths
        self._arrow_group = VGroup()
        for dot in self.object_circles:
            end = dot.get_top()
            arrow = Arrow(fan_origin, end, buff=0.06, stroke_width=1.6,
                          color=arrow_color, max_tip_length_to_length_ratio=0.04,
                          tip_length=0.14)
            arrow.set_opacity(0.28)
            self.arrows.append(arrow)
            self.arrow_paths.append(Line(end, fan_origin))
            self._arrow_group.add(arrow)

        self.add(self._arrow_group, self.room_node, self.objects)

    @staticmethod
    def _spread(n, width):
        if n == 1:
            return [0.0]
        return list(np.linspace(-width / 2, width / 2, n))

    # ── accessors ───────────────────────────────────────────────────────────
    def all_arrows(self):
        return self._arrow_group

    # ── posterior bars ───────────────────────────────────────────────────────
    def set_posterior(self, values, highlight=None, color=None):
        """Animate the in-node bar chart to `values` (length = #rooms).

        Heights are scaled so the largest value fills `bar_max_h`. If
        `highlight` (a room index) is given, that bar is recolored.
        """
        values = np.asarray(values, dtype=float)
        scale = self.bar_max_h / max(values.max(), 1e-9)
        color = color or GREEN
        anims = []
        for i, bar in enumerate(self.bars):
            h = max(0.04, values[i] * scale)
            x = self.bar_xs[i]
            a = bar.animate.stretch_to_fit_height(h).move_to(
                [x, self.bar_baseline_y + h / 2, 0])
            if highlight is not None:
                a = a.set_fill(color if i == highlight else self.room_color,
                               opacity=0.95 if i == highlight else 0.4)
            anims.append(a)
        self._bar_values = values
        return AnimationGroup(*anims)

    # ── evidence on object nodes ──────────────────────────────────────────────
    def mark_present(self, idx, color=GREEN):
        dot, txt = self.objects[idx]
        check = theme.crisp_text("✓", font_size=20, color=color).move_to(dot)
        self._tag(dot, check)
        return AnimationGroup(
            dot.animate.set_color(color).set_fill(color, opacity=0.45),
            txt.animate.set_color(color),
            FadeIn(check, scale=0.6),
        )

    def mark_absent(self, idx, color=RED):
        dot, txt = self.objects[idx]
        cross = Cross(dot, stroke_color=color, stroke_width=4).scale(0.55)
        self._tag(dot, cross)
        return AnimationGroup(
            dot.animate.set_stroke(color, opacity=0.5).set_fill(color, opacity=0.0),
            txt.animate.set_color(color).set_opacity(0.55),
            FadeIn(cross),
        )

    def _tag(self, dot, mob):
        # keep evidence marks so callers can fade them with the graph
        if not hasattr(self, "_marks"):
            self._marks = VGroup()
            self.add(self._marks)
        self._marks.add(mob)

    def pulse(self, indices, color=YELLOW, run_time=0.9):
        """Dots travelling UP the arrows (object -> room): evidence flowing in.

        Returns (dots, animation); the caller should add `dots`, play the
        animation, then remove `dots`.
        """
        dots = VGroup()
        anims = []
        for i in indices:
            d = Dot(color=color, radius=0.07).move_to(self.object_circles[i].get_top())
            dots.add(d)
            anims.append(MoveAlongPath(d, self.arrow_paths[i], run_time=run_time))
        return dots, AnimationGroup(*anims, lag_ratio=0.04)

    # ── conditional-independence illustration ────────────────────────────────
    def independence_demo(self, pairs=None, color=RED):
        """Build arcs between object pairs + crosses over them.

        Returns (edges, crosses): the caller fades them in to suggest
        object↔object links, then crosses/fades them out — "we do NOT model
        these dependencies". By default links a few adjacent pairs.
        """
        n = len(self.object_circles)
        if pairs is None:
            pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
        edges = VGroup()
        crosses = VGroup()
        for a, b in pairs:
            ca, cb = self.object_circles[a], self.object_circles[b]
            arc = ArcBetweenPoints(ca.get_top() + UP * 0.05,
                                   cb.get_top() + UP * 0.05,
                                   angle=-PI / 3, color=color, stroke_width=3)
            edges.add(arc)
            # fixed-size, square X at the arc's midpoint (a bbox-scaled Cross
            # over the wide/flat arc would render as two near-horizontal lines)
            x = Cross(stroke_color=color, stroke_width=5).scale(0.13)
            x.move_to(arc.point_from_proportion(0.5))
            crosses.add(x)
        return edges, crosses
