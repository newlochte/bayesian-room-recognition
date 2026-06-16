"""
Scene 7 – Demo: live inference on three of our own photos.

One persistent "Demo" title. For every scene in output/ (salon, sypialnia,
schody) we rebuild the inference picture straight from its results.json:

    ┌────────────────────┬───────────────────────────┐
    │                    │  P(pomieszczenie|obiekty)  │  ← real posteriors
    │   photo + YOLOv8   ├───────────────────────────┤
    │   bounding boxes   │   bayesian network view    │  ← room -> objects,
    └────────────────────┴───────────────────────────┘     edges = P(obj|room)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
import theme
import json

REPO   = os.path.join(os.path.dirname(__file__), "../..")
BORM   = os.path.join(REPO, "models/borm")

# scenes to show, in order: (output folder, ground-truth caption in PL)
SCENES = [
    ("salon",     "salon"),
    ("sypialnia", "sypialnia"),
    ("schody",    "schody"),
]

ROOM_PL = {
    "office": "biuro", "bedroom": "sypialnia", "kitchen": "kuchnia",
    "bathroom": "łazienka", "living_room": "salon", "dining_room": "jadalnia",
    "corridor": "korytarz", "staircase": "schody", "garage-indoor": "garaż",
    "gymnasium-indoor": "siłownia", "library-indoor": "biblioteka",
}
OBJ_PL = {
    "dining table": "stół", "chair": "krzesło", "bowl": "miska",
    "couch": "kanapa", "cup": "kubek", "bed": "łóżko", "tv": "telewizor",
    "keyboard": "klawiatura", "vase": "wazon", "potted plant": "roślina",
    "oven": "piekarnik", "microwave": "mikrofalówka", "book": "książka",
    "laptop": "laptop", "refrigerator": "lodówka", "sink": "zlew",
    "toilet": "toaleta", "person": "osoba",
}
PALETTE = [GREEN, BLUE_C, ORANGE, PURPLE_B, RED_C, TEAL_C, GOLD_D, PINK,
           MAROON_C, LIGHT_BROWN]


def load_model():
    """P(object|room) table + name lists, for weighting network edges."""
    try:
        P = np.load(os.path.join(BORM, "P_object_given_room.npy"))
        rooms = json.load(open(os.path.join(BORM, "room_names.json")))
        objs  = json.load(open(os.path.join(BORM, "object_names.json")))
        return P, rooms, objs
    except Exception:
        return None, None, None


class SceneDemo(Scene):
    def construct(self):
        self.P, self.room_names, self.object_names = load_model()

        title = theme.crisp_text("Demo", font_size=46, color=theme.TITLE)
        title.to_edge(UP, buff=0.32)
        self.play(Write(title), run_time=0.7)

        for i, (folder, caption) in enumerate(SCENES):
            self._demo(folder, caption)
            if i < len(SCENES) - 1:
                self.wait(0.3)

        self.play(FadeOut(title))

    # ── one inference panel ───────────────────────────────────────────────────
    def _demo(self, folder, caption):
        data = json.load(open(os.path.join(REPO, "output", folder, "results.json")))
        img_path = os.path.join(REPO, data["image"])

        content = Group()  # everything except the persistent title

        # ── LEFT: photo + bounding boxes ───────────────────────────────────────
        img = ImageMobject(img_path)
        img.set_height(5.4)
        if img.width > 6.0:
            img.set_width(6.0)
        img.move_to(LEFT * 3.55 + DOWN * 0.35)
        frame = SurroundingRectangle(img, color=theme.MUTED, buff=0,
                                     stroke_width=1.5)

        cap = theme.crisp_text(f"{caption}  ·  YOLOv8", font_size=20,
                               color=theme.MUTED)
        cap.next_to(img, UP, buff=0.16).align_to(img, LEFT)

        self.play(FadeIn(img), Create(frame), FadeIn(cap, shift=DOWN * 0.1),
                  run_time=0.7)
        content.add(img, frame, cap)

        rects, tags = self._boxes(img, data["detections"])
        box_group = VGroup(*rects, *tags)
        self.play(LaggedStart(*[Create(r) for r in rects],
                              lag_ratio=0.18, run_time=1.4))
        self.play(LaggedStart(*[FadeIn(t, scale=0.8) for t in tags],
                              lag_ratio=0.18, run_time=0.8))
        content.add(box_group)
        self.wait(0.3)

        # ── TOP RIGHT: posterior over rooms ────────────────────────────────────
        post = data["posterior"]
        top5 = sorted(post.items(), key=lambda x: -x[1])[:5]
        pred = data["predicted_room"]

        prob_head = theme.crisp_text("P(pomieszczenie | obiekty)",
                                     font_size=20, color=theme.ACCENT_LIGHT)
        bars = self._prob_bars(top5, pred)
        prob_block = VGroup(prob_head, bars).arrange(DOWN, buff=0.28,
                                                     aligned_edge=LEFT)
        prob_block.move_to(RIGHT * 3.7 + UP * 2.0)
        self.play(FadeIn(prob_head, shift=DOWN * 0.1), run_time=0.4)
        for row in bars.submobjects:
            self.play(GrowFromEdge(row, LEFT), run_time=0.28)
        content.add(prob_block)
        self.wait(0.3)

        # ── BOTTOM RIGHT: bayesian network view ────────────────────────────────
        net_head = theme.crisp_text("Sieć bayesowska", font_size=20,
                                    color=theme.ACCENT_LIGHT)
        net_head.move_to(RIGHT * 3.2 + UP * 0.1)
        net = self._network(data, center=RIGHT * 3.2 + DOWN * 1.95)
        self.play(FadeIn(net_head, shift=DOWN * 0.1), run_time=0.4)
        net.animate_in(self)
        content.add(net_head, net)
        self.wait(1.3)

        self.play(FadeOut(content))
        self.clear_objects(content)

    def clear_objects(self, group):
        for m in list(group):
            self.remove(m)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _boxes(self, img, detections):
        """Build pixel-accurate rectangles; label the top box of each class."""
        PH, PW = img.pixel_array.shape[0], img.pixel_array.shape[1]
        ul = img.get_corner(UL)
        W, H = img.width, img.height

        classes = []
        for d in detections:
            if d["name"] not in classes:
                classes.append(d["name"])
        class_color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}

        rects, tags, tagged = [], [], set()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            mx1 = ul[0] + (x1 / PW) * W
            mx2 = ul[0] + (x2 / PW) * W
            mty = ul[1] - (y1 / PH) * H
            mby = ul[1] - (y2 / PH) * H
            col = class_color[d["name"]]
            rect = Rectangle(width=mx2 - mx1, height=mty - mby,
                             color=col, stroke_width=2.5)
            rect.move_to([(mx1 + mx2) / 2, (mty + mby) / 2, 0])
            rects.append(rect)

            if d["name"] not in tagged:  # one tidy label per class
                tagged.add(d["name"])
                label = OBJ_PL.get(d["name"], d["name"])
                txt = theme.crisp_text(f"{label} {int(d['confidence'] * 100)}%",
                                       font_size=14, color=BLACK)
                bg = BackgroundRectangle(txt, color=col, fill_opacity=0.9,
                                         buff=0.05)
                tag = VGroup(bg, txt)
                tag.next_to(rect.get_corner(UL), DOWN, buff=0).align_to(
                    rect, LEFT)
                tags.append(tag)
        return rects, tags

    def _prob_bars(self, items, pred, max_w=2.4, bar_h=0.32, gap=0.16,
                   label_w=1.6):
        rows = VGroup()
        for room, p in items:
            is_pred = room == pred
            col = theme.HIGHLIGHT if is_pred else theme.ACCENT
            lbl = theme.crisp_text(ROOM_PL.get(room, room), font_size=16,
                                   color=WHITE if is_pred else theme.MUTED)
            lbl.set(width=min(lbl.width, label_w))
            bg = Rectangle(width=max_w, height=bar_h, fill_color=DARK_GRAY,
                           fill_opacity=0.45, stroke_width=0)
            fill = Rectangle(width=max(max_w * p, 0.012), height=bar_h,
                             fill_color=col, fill_opacity=0.95, stroke_width=0)
            fill.align_to(bg, LEFT)
            pct = theme.crisp_text(f"{p * 100:.1f}%", font_size=14,
                                   color=theme.MUTED)
            lbl.next_to(bg, LEFT, buff=0.18)
            pct.next_to(bg, RIGHT, buff=0.14)
            rows.add(VGroup(lbl, bg, fill, pct))
        rows.arrange(DOWN, buff=gap, aligned_edge=LEFT)
        # keep the bar backgrounds vertically aligned (labels vary in width)
        for r in rows:
            r[1].align_to(rows[0][1], LEFT)
            r[2].align_to(rows[0][1], LEFT)
            r[3].next_to(r[1], RIGHT, buff=0.14)
            r[0].next_to(r[1], LEFT, buff=0.18)
        return rows

    def _network(self, data, center):
        return _StarNetwork(data["predicted_room"], data["detected_objects"],
                            self.P, self.room_names, self.object_names,
                            center=center)


# bayes_network.png colour conventions
ROOM_FILL, ROOM_EDGE = "#fefcbf", "#b7791f"
OBS_FILL,  OBS_EDGE  = "#bee3f8", "#2b6cb0"
UNOBS_EDGE, UNOBS_ARROW = "#4a5568", "#a0aec0"


class _StarNetwork(VGroup):
    """Room -> objects star network, styled like output/<room>/bayes_network.png.

    A hidden 'Room' node (yellow) on top fans down to object nodes: the detected
    ones are drawn filled blue (observed evidence), the rest stay white
    (unobserved). Each directed edge is labelled with P(object | predicted room)
    from the trained model. Objects shown = detections first, then the most
    informative remaining ones, capped so the row stays tight.
    """

    MAX_OBJECTS = 6

    def __init__(self, pred_room, detected, P, room_names, object_names,
                 center=ORIGIN, width=5.6, **kwargs):
        super().__init__(**kwargs)
        cx, cy = center[0], center[1]
        detected = list(detected)

        chosen = self._choose_objects(detected, P, object_names)
        room_idx = (room_names.index(pred_room)
                    if P is not None and pred_room in room_names else None)

        # ── room node (hidden variable we infer) ────────────────────────────
        room_y = cy + 1.0
        rbox = Ellipse(width=2.0, height=0.98, color=ROOM_EDGE,
                       fill_color=ROOM_FILL, fill_opacity=1.0, stroke_width=3)
        rbox.move_to([cx, room_y, 0])
        rtxt = theme.crisp_text(f"Pomieszczenie\n({ROOM_PL.get(pred_room, pred_room)})",
                                font_size=15, color=BLACK)
        if rtxt.width > 1.7:
            rtxt.set(width=1.7)
        rtxt.move_to(rbox)
        self.room_node = VGroup(rbox, rtxt)

        # ── object nodes (observed = blue, unobserved = white) ──────────────
        obj_y = cy - 0.7
        n = len(chosen)
        xs = [cx] if n <= 1 else list(np.linspace(cx - width / 2, cx + width / 2, n))
        self.obj_nodes, self.edges, self.edge_labels = VGroup(), VGroup(), VGroup()
        for x, o in zip(xs, chosen):
            observed = o in detected
            ell = Ellipse(width=0.92, height=0.62,
                          fill_color=OBS_FILL if observed else "#ffffff",
                          fill_opacity=1.0,
                          color=OBS_EDGE if observed else UNOBS_EDGE,
                          stroke_width=2)
            ell.move_to([x, obj_y, 0])
            lbl = theme.crisp_text(OBJ_PL.get(o, o), font_size=13, color=BLACK)
            if lbl.width > 0.82:
                lbl.set(width=0.82)
            lbl.move_to(ell)
            self.obj_nodes.add(VGroup(ell, lbl))

            arrow = Arrow(rbox.get_bottom(), ell.get_top(), buff=0.06,
                          stroke_width=2.4 if observed else 1.4,
                          color=OBS_EDGE if observed else UNOBS_ARROW,
                          max_tip_length_to_length_ratio=0.12, tip_length=0.13)
            self.edges.add(arrow)

            if room_idx is not None:
                p = float(P[room_idx, object_names.index(o)])
                ptxt = theme.crisp_text(f"{p:.2f}", font_size=16, weight=BOLD,
                                        color=OBS_FILL if observed else "#cbd5e0")
                ptxt.move_to(arrow.get_center())
                ptxt.add_to_back(BackgroundRectangle(ptxt, color=BLACK,
                                                     fill_opacity=0.9, buff=0.05))
                self.edge_labels.add(ptxt)

        # ── compact legend ──────────────────────────────────────────────────
        def chip(fill, edge, text):
            sw = Ellipse(width=0.22, height=0.16, fill_color=fill,
                         fill_opacity=1.0, color=edge, stroke_width=1.5)
            tx = theme.crisp_text(text, font_size=12, color=theme.MUTED)
            return VGroup(sw, tx).arrange(RIGHT, buff=0.08)
        self.legend = VGroup(
            chip(OBS_FILL, OBS_EDGE, "obserwowany"),
            chip("#ffffff", UNOBS_EDGE, "nieobserwowany"),
        ).arrange(RIGHT, buff=0.4)
        self.legend.move_to([cx, obj_y - 0.78, 0])

        self.add(self.edges, self.edge_labels, self.room_node,
                 self.obj_nodes, self.legend)

    @classmethod
    def _choose_objects(cls, detected, P, object_names):
        chosen = [o for o in detected if object_names and o in object_names]
        if P is not None and object_names:
            best = P.max(axis=0)
            for j in np.argsort(best)[::-1]:
                if len(chosen) >= cls.MAX_OBJECTS:
                    break
                name = object_names[j]
                if name not in chosen:
                    chosen.append(name)
        return chosen[:cls.MAX_OBJECTS]

    def animate_in(self, scene):
        scene.play(FadeIn(self.room_node, shift=DOWN * 0.15), run_time=0.5)
        scene.play(LaggedStart(*[FadeIn(o, shift=UP * 0.15)
                                 for o in self.obj_nodes],
                               lag_ratio=0.18, run_time=0.8))
        scene.play(
            LaggedStart(*[GrowArrow(e) for e in self.edges],
                        lag_ratio=0.12, run_time=0.8),
            LaggedStart(*[FadeIn(t) for t in self.edge_labels],
                        lag_ratio=0.12, run_time=0.8),
        )
        scene.play(FadeIn(self.legend), run_time=0.4)
