import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
import theme


class BayesBipartiteGraph(VGroup):
    """
    Bipartite Naive-Bayes view:

        room hypotheses  (top row, rounded boxes)
              ⟷
        detected objects (bottom row, circles)

    Every room connects to every object — in Naive Bayes each object is a
    child of the room class. Edges are faint by default; the helper methods
    brighten a room (and its edges) to show which hypothesis the evidence
    supports, or mark individual objects/edges as present/absent.
    """

    def __init__(self, room_labels, object_labels,
                 width=10.5, object_width=None, room_y=1.9, object_y=-1.9,
                 room_color=None, object_color=None, edge_color=None,
                 room_w=1.9, room_h=0.72, object_r=0.45,
                 font_size_room=20, font_size_object=20,
                 edge_stroke=1.4, edge_opacity=0.25, **kwargs):
        super().__init__(**kwargs)
        self.room_color   = room_color   or theme.BAYES_CENTER
        self.object_color = object_color or theme.BAYES_LEAF
        edge_color        = edge_color   or theme.EDGE
        # The object row can be spread wider than the rooms so a long vocabulary
        # of objects still fits with readable gaps.
        object_width = object_width if object_width is not None else width * 0.9

        # ── room nodes (top row) ───────────────────────────────────────────
        self.rooms = VGroup()
        self.room_boxes = []
        for x, label in zip(self._spread(len(room_labels), width), room_labels):
            box = RoundedRectangle(corner_radius=0.12, width=room_w, height=room_h,
                                   color=self.room_color, fill_color=self.room_color,
                                   fill_opacity=0.18, stroke_width=2.5)
            box.move_to([x, room_y, 0])
            txt = theme.crisp_text(label, font_size=font_size_room,
                                   color=self.room_color)
            if txt.width > room_w - 0.2:
                txt.scale((room_w - 0.2) / txt.width)
            txt.move_to(box)
            self.room_boxes.append(box)
            self.rooms.add(VGroup(box, txt))

        # ── object nodes (bottom row) ──────────────────────────────────────
        self.objects = VGroup()
        self.object_circles = []
        for x, label in zip(self._spread(len(object_labels), object_width), object_labels):
            dot = Circle(radius=object_r, color=self.object_color,
                         fill_color=self.object_color, fill_opacity=0.15,
                         stroke_width=2)
            dot.move_to([x, object_y, 0])
            txt = theme.crisp_text(label, font_size=font_size_object,
                                   color=self.object_color)
            txt.next_to(dot, DOWN, buff=0.14)
            self.object_circles.append(dot)
            self.objects.add(VGroup(dot, txt))

        # ── edges: edges[r][o] joins room r to object o ────────────────────
        self.edges = [[None] * len(object_labels) for _ in room_labels]
        self._edge_group = VGroup()
        for r, rbox in enumerate(self.room_boxes):
            for o, odot in enumerate(self.object_circles):
                edge = Line(rbox.get_bottom(), odot.get_top(),
                            color=edge_color, stroke_width=edge_stroke,
                            stroke_opacity=edge_opacity)
                self.edges[r][o] = edge
                self._edge_group.add(edge)

        self.add(self._edge_group, self.rooms, self.objects)

    @staticmethod
    def _spread(n, width):
        if n == 1:
            return [0.0]
        return list(np.linspace(-width / 2, width / 2, n))

    # ── animation helpers ──────────────────────────────────────────────────
    def all_edges(self):
        return self._edge_group

    def room_edges(self, r):
        return VGroup(*self.edges[r])

    def highlight_room(self, r, color=None, edge_width=3.0):
        """Light up a room hypothesis and all of its edges."""
        color = color or self.room_color
        box, txt = self.rooms[r]
        anims = [
            box.animate.set_stroke(color, width=3.5).set_fill(color, opacity=0.4),
            txt.animate.set_color(color),
        ]
        for edge in self.edges[r]:
            anims.append(edge.animate.set_color(color)
                         .set_stroke(width=edge_width, opacity=1.0))
        return AnimationGroup(*anims)

    def emphasize_room(self, r, weights, color=None,
                       width_range=(0.8, 4.4), opacity_range=(0.12, 1.0),
                       fill_range=(0.04, 0.6)):
        """Light up a room and scale every edge + object by its contribution.

        weights[o] in [0, 1] is how strongly object o supports room r. Higher
        weight → a thicker, more opaque edge and a more vivid object node, so a
        glance shows which objects actually drive the decision.
        """
        color = color or self.room_color
        box, txt = self.rooms[r]
        anims = [
            box.animate.set_stroke(color, width=3.5).set_fill(color, opacity=0.45),
            txt.animate.set_color(color),
        ]
        (w0, w1), (o0, o1), (f0, f1) = width_range, opacity_range, fill_range
        for o, w in enumerate(weights):
            w = max(0.0, min(1.0, w))
            anims.append(self.edges[r][o].animate.set_color(color).set_stroke(
                width=w0 + (w1 - w0) * w, opacity=o0 + (o1 - o0) * w))

            dot, dtxt = self.objects[o]
            node_stroke = interpolate_color(self.object_color, color, w)
            anims.append(dot.animate
                         .set_stroke(node_stroke, width=2 + 1.6 * w, opacity=0.35 + 0.65 * w)
                         .set_fill(color, opacity=f0 + (f1 - f0) * w))
            anims.append(dtxt.animate.set_color(interpolate_color(GREY_B, color, w)))
        return AnimationGroup(*anims)

    def fade_room(self, r, opacity=0.22):
        """Dim a rejected room hypothesis and its edges into the background."""
        box, txt = self.rooms[r]
        anims = [box.animate.set_opacity(opacity), txt.animate.set_opacity(opacity)]
        for edge in self.edges[r]:
            anims.append(edge.animate.set_stroke(opacity=0.1))
        return AnimationGroup(*anims)

    def mark_object(self, o, color, fill=0.4):
        dot, txt = self.objects[o]
        return AnimationGroup(
            dot.animate.set_color(color).set_fill(color, opacity=fill),
            txt.animate.set_color(color),
        )

    def highlight_edge(self, r, o, color, width=3.5):
        return self.edges[r][o].animate.set_color(color).set_stroke(
            width=width, opacity=1.0)
