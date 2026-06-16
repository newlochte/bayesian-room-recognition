"""
Scene 3+4 – YOLO detection morphing into the Bayesian network (merged scene).
Phase 1: Image + all bounding boxes appear at once (names only, no confidence).
Phase 2: Cards (crop + colored border) fly to a centered list; image vanishes.
Phase 3: Object names appear next to thumbnails, then confidence scores fade in.
Phase 4: Detections below the 60% confidence threshold are dropped.
Phase 5: The surviving cards morph into the leaf nodes of a Bayesian star graph,
         then the rest of the graph grows in.
Phase 6: Naive Bayes formula.
Phase 7: Edge-weight animation (presence / absence as evidence).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
from manim import *
import theme
from components.bayes_star import BayesStarGraph, naive_bayes_posterior
from PIL import Image
import tempfile

IMG_DIR = os.path.join(os.path.dirname(__file__), "../assets/images")

PALETTE = [GREEN_D, GREEN_C, GREEN_B, GREEN_A, YELLOW, RED]

# Candidate room hypotheses shown along the top of the Bayesian graph. The
# detected objects (krzesło / monitor / klawiatura / mysz) point at "Biuro".
ROOMS = ["Kuchnia", "Salon", "Sypialnia", "Biuro", "Łazienka"]

# Abstract P(object | room) priors in [0, 1]: how strongly each object suggests
# a room. Only the relative values matter — they drive how bright/thick each
# node and edge becomes when its room is emphasised. Missing pairs default low.
ROOM_OBJECT_WEIGHT = {
    "Kuchnia":   {"kuchenka": .95, "zlew": .88, "stół": .60, "krzesło": .42,
                  "kanapa": .12, "toaleta": .14},
    "Salon":     {"kanapa": .92, "stół": .55, "krzesło": .52, "monitor": .45,
                  "łóżko": .12, "klawiatura": .12, "mysz": .12},
    "Sypialnia": {"łóżko": .95, "kanapa": .38, "krzesło": .30, "stół": .24,
                  "monitor": .20, "toaleta": .16},
    "Biuro":     {"monitor": .95, "klawiatura": .90, "mysz": .88, "krzesło": .80,
                  "stół": .50, "kanapa": .30},
    "Łazienka":  {"toaleta": .95, "zlew": .85, "krzesło": .15, "stół": .12,
                  "kanapa": .08},
}


def _banner(text, color, font_size=30, y=-0.1):
    """Caption centered on screen with a dark backing so it stays legible over
    the arrows of the star graph."""
    txt = theme.crisp_text_arrow(text, font_size=font_size, color=color)
    bg = RoundedRectangle(corner_radius=0.1, width=txt.width + 0.5,
                          height=txt.height + 0.3, fill_color=BLACK,
                          fill_opacity=0.78, stroke_width=0)
    bg.move_to([0, y, 0])
    txt.move_to(bg)
    return VGroup(bg, txt)


def _bbox_crop(img_path, box, pad=0.05):
    """Crop the bounding box region with a small padding border."""
    img = Image.open(img_path)
    W, H = img.size
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    crop = img.crop((
        max(0, int(x1 - pw)), max(0, int(y1 - ph)),
        min(W, int(x2 + pw)), min(H, int(y2 + ph)),
    ))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    crop.save(tmp.name)
    return tmp.name


class SceneYoloBayes(Scene):
    def construct(self):
        self._yolo()

    # ── Phases 1–5: YOLO detection → morph into the star graph ─────────────
    def _yolo(self):
        img_path = os.path.join(IMG_DIR, "office.jpg")
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size

        # Scale: pixels → manim units. sx == sy because image is at natural aspect.
        display_h = 5.0
        display_w = img_w / img_h * display_h
        sx = display_w / img_w  # == display_h / img_h

        dets = [
                {"name": "krzesło",     "confidence": 0.9160, "box": [1761.65, 1600.33, 3151.78, 3612.54]},
                {"name": "monitor",     "confidence": 0.8676, "box": [3264.44, 1752.59, 4147.14, 2306.48]},
                {"name": "klawiatura",  "confidence": 0.7968, "box": [2837.23, 2284.88, 3525.48, 2509.80]},
                {"name": "mysz",        "confidence": 0.7493, "box": [3473.18, 2451.53, 3664.79, 2550.68]},
                {"name": "samochód",    "confidence": 0.5058, "box": [2835.57, 1821.40, 3120.27, 1994.21]},
                {"name": "laptop",      "confidence": 0.2900, "box": [4121.05, 1631.34, 5226.07, 2604.59]},
            ]

        # ── Phase 1: full image ────────────────────────────────────────────
        img = ImageMobject(img_path).scale_to_fit_height(display_h).move_to(ORIGIN)
        title = theme.crisp_text("Detekcja obiektów – YOLO", font_size=40, color=theme.TITLE)
        title.to_edge(UP, buff=0.5)

        self.play(FadeIn(title, shift=DOWN * 0.4), run_time=0.7)
        self.play(FadeIn(img), run_time=0.9)
        self.wait(0.3)

        # ── Build cards at bbox positions ──────────────────────────────────
        cards = []      # Group(crop_img, border_rect)
        name_tags = []  # label on the box  – name only, no confidence
        box_dims = []   # (bw, bh, cx, cy) in manim units

        for i, det in enumerate(dets):
            color = PALETTE[i % len(PALETTE)]
            x1, y1, x2, y2 = det["box"]

            bw = (x2 - x1) * sx
            bh = (y2 - y1) * sx          # sx == sy
            cx = x1 * sx - display_w / 2 + bw / 2
            cy = display_h / 2 - y1 * sx - bh / 2

            # Crop image: set_height(bh) gives width=bw automatically (square pixels).
            # Starts invisible so Phase 1 shows the real photo through the border;
            # the crop is revealed only when the fly-out movement begins.
            cpath = _bbox_crop(img_path, det["box"])
            cimg = ImageMobject(cpath).scale_to_fit_height(bh).move_to(np.array([cx, cy, 0.0]))
            cimg.set_opacity(0)

            border = Rectangle(width=bw, height=bh, color=color, stroke_width=2.5)
            border.move_to([cx, cy, 0])

            # Name tag at top-left corner of box
            tag_txt = theme.crisp_text(det["name"], font_size=15, color=BLACK, weight=BOLD)
            tag_bg = Rectangle(
                width=tag_txt.width + 0.12, height=tag_txt.height + 0.08,
                fill_color=color, fill_opacity=0.92, stroke_width=0,
            )
            tag_bg.align_to(border, UL).shift(UP * 0.02)
            tag_txt.move_to(tag_bg)

            cards.append(Group(cimg, border))
            name_tags.append(VGroup(tag_bg, tag_txt))
            box_dims.append((bw, bh, cx, cy))

        # All boxes appear at once over the photo — only borders + name tags
        # (the cropped thumbnails inside each card are still invisible here).
        self.add(*[c[0] for c in cards])
        self.play(
            *[FadeIn(c[1]) for c in cards],
            *[FadeIn(t) for t in name_tags],
            run_time=0.6,
        )
        self.wait(0.7)

        # ── Phase 2: cards fly to a centered 2-column grid; image vanishes ──
        THUMB_H   = 1.0     # bigger final thumbnails than before
        MAX_W     = 1.5
        ROW_GAP   = 0.45
        COL_GAP   = 1.0
        NAME_GAP  = 0.30    # thumbnail → name
        CONF_GAP  = 0.28    # name → confidence
        NCOLS     = 2

        n = len(dets)

        # Scale factor per card (uniform scale, capped so no card is too wide)
        scale_factors = []
        for bw, bh, *_ in box_dims:
            f = THUMB_H / bh
            if bw * f > MAX_W:
                f = MAX_W / bw
            scale_factors.append(f)

        # Build name + confidence labels up front so we can measure their widths.
        name_labels = []
        conf_labels = []
        for i, det in enumerate(dets):
            color = PALETTE[i % len(PALETTE)]
            name_labels.append(theme.crisp_text(det["name"], font_size=28, color=color))
            conf_pct = int(det["confidence"] * 100)
            conf_labels.append(theme.crisp_text(f"{conf_pct}%", font_size=24, color=theme.MUTED))

        # Uniform cell sizing so both columns line up.
        max_thumb_w = max(box_dims[i][0] * scale_factors[i] for i in range(n))
        max_name_w  = max(nl.width for nl in name_labels)
        max_conf_w  = max(cl.width for cl in conf_labels)
        cell_w = max_thumb_w + NAME_GAP + max_name_w + CONF_GAP + max_conf_w

        nrows = (n + NCOLS - 1) // NCOLS
        total_w = NCOLS * cell_w + (NCOLS - 1) * COL_GAP
        total_h = nrows * THUMB_H + (nrows - 1) * ROW_GAP

        # Per-column / per-row anchors, with the whole grid centered on screen.
        col_left_x  = [-total_w / 2 + c * (cell_w + COL_GAP) for c in range(NCOLS)]
        thumb_x     = [cx + max_thumb_w / 2 for cx in col_left_x]
        name_left_x = [cx + max_thumb_w + NAME_GAP for cx in col_left_x]
        row_y = [total_h / 2 - THUMB_H / 2 - r * (THUMB_H + ROW_GAP) for r in range(nrows)]

        # Row-major fill: item i → (row i//NCOLS, col i%NCOLS). The two lowest
        # confidence dets land in the last row, so dropping them leaves a tidy grid.
        cell_rc = [(i // NCOLS, i % NCOLS) for i in range(n)]
        target_xy = [(thumb_x[c], row_y[r]) for (r, c) in cell_rc]

        # Place the (still hidden) labels at their final spots.
        for i in range(n):
            r, c = cell_rc[i]
            name_labels[i].move_to([0.0, row_y[r], 0.0]).align_to([name_left_x[c], 0.0, 0.0], LEFT)
            conf_labels[i].next_to(name_labels[i], RIGHT, buff=CONF_GAP)

        # Reveal the cropped thumbnails just as the movement begins, so each box
        # switches from "window onto the photo" to the card it carries away.
        for c in cards:
            c[0].set_opacity(1)
        # Names are known straight from the image, so they fade in together
        # with the move and are present the moment each card arrives.
        self.play(
            *[
                cards[i].animate.scale(scale_factors[i]).move_to([*target_xy[i], 0])
                for i in range(n)
            ],
            *[FadeIn(nl) for nl in name_labels],
            *[FadeOut(t) for t in name_tags],
            FadeOut(img),
            FadeOut(title),
            run_time=1.2,
        )
        self.wait(0.3)

        # ── Phase 3: confidence scores fade in afterwards ──────────────────
        self.play(
            LaggedStart(*[FadeIn(cl) for cl in conf_labels], lag_ratio=0.12, run_time=0.7)
        )
        self.wait(1.5)

        # ── Phase 4: drop detections below the 60% confidence threshold ────
        THRESHOLD = 0.60
        low_idx  = [i for i, d in enumerate(dets) if d["confidence"] <  THRESHOLD]
        keep_idx = [i for i, d in enumerate(dets) if d["confidence"] >= THRESHOLD]

        thr_caption = theme.Text(
            "Odrzucamy detekcje z pewnością poniżej 60%",
            font_size=28, color=RED,
        )
        thr_caption.to_edge(UP, buff=0.5)
        self.play(FadeIn(thr_caption, shift=DOWN * 0.15))
        self.wait(0.8)

        # Recenter the surviving rows vertically as the rejected ones fade away.
        keep_rows = sorted({i // NCOLS for i in keep_idx})
        new_nrows = len(keep_rows)
        new_total_h = new_nrows * THUMB_H + (new_nrows - 1) * ROW_GAP
        new_row_y = {r: new_total_h / 2 - THUMB_H / 2 - k * (THUMB_H + ROW_GAP)
                     for k, r in enumerate(keep_rows)}
        dy = {i: new_row_y[i // NCOLS] - row_y[i // NCOLS] for i in keep_idx}

        self.play(
            *[FadeOut(cards[i], shift=RIGHT * 0.4) for i in low_idx],
            *[FadeOut(name_labels[i], shift=RIGHT * 0.4) for i in low_idx],
            *[FadeOut(conf_labels[i], shift=RIGHT * 0.4) for i in low_idx],
            *[
                Group(cards[i], name_labels[i], conf_labels[i])
                .animate.shift(UP * dy[i])
                for i in keep_idx
            ],
            run_time=1.0,
        )
        self.wait(1.2)

        self.play(FadeOut(thr_caption))
        self.wait(0.6)

        # ── Phase 5: surviving cards morph into the Bayesian STAR network ───
        # The network's true structure: ONE room variable, one directed arrow to
        # each object. The first four object nodes are the surviving YOLO
        # detections (confidence order) that the cards morph into 1:1; the rest
        # are other vocabulary objects that grow in afterwards.
        detected = [dets[i]["name"] for i in keep_idx]   # confidence order
        extra_objects = ["łóżko", "kanapa", "stół", "zlew", "toaleta", "kuchenka"]
        all_objects = detected + extra_objects
        # No section title here (it crowded the room node); node sits a bit lower.
        graph = BayesStarGraph(ROOMS, all_objects, object_width=12.0, node_y=2.15)

        # keep_idx is in confidence order (krzesło, monitor, klawiatura, mysz),
        # matching the first four graph.objects 1:1. cards[i] is a Group(thumbnail,
        # border): the image can't be morphed (ImageMobject has no color
        # interpolation), so we fade it out and morph the VMobjects:
        # border → object circle, name → label.
        morph = []
        for slot, i in enumerate(keep_idx):
            obj_dot, obj_txt = graph.objects[slot]
            morph.append(FadeOut(cards[i][0]))                          # thumbnail
            morph.append(FadeOut(conf_labels[i]))                       # confidence
            morph.append(ReplacementTransform(cards[i][1], obj_dot))    # border → circle
            morph.append(ReplacementTransform(name_labels[i], obj_txt)) # name → label

        self.play(*morph, run_time=1.4)
        self.wait(0.3)

        # The rest of the known objects grow in beside the detected ones.
        rest_objs = range(len(detected), len(all_objects))
        self.play(
            LaggedStart(*[FadeIn(graph.objects[o], shift=UP * 0.2) for o in rest_objs],
                        lag_ratio=0.1),
            run_time=1.1,
        )
        self.wait(0.2)

        # The single room node drops in along the top (uniform belief so far).
        self.play(FadeIn(graph.room_node, shift=DOWN * 0.25), run_time=0.9)
        self.wait(1.2)

        # Directed edges: each object is a child of the one room variable.
        self.play(LaggedStart(*[GrowArrow(a) for a in graph.arrows],
                              lag_ratio=0.04), run_time=1.1)
        self.wait(1.8)

        # The "naive" assumption, drawn explicitly: candidate object↔object
        # links appear, then get crossed out — we do NOT model them.
        indep_edges, indep_crosses = graph.independence_demo()
        indep_lines = VGroup(
            theme.crisp_text("Brak krawędzi między obiektami",
                             font_size=30, color=theme.TITLE),
            MathTex(r"\downarrow", color=theme.TITLE).scale(1.1),
            theme.crisp_text("niezależność", font_size=30, color=theme.TITLE),
        ).arrange(DOWN, buff=0.14)
        indep_bg = RoundedRectangle(
            corner_radius=0.1, width=indep_lines.width + 0.5,
            height=indep_lines.height + 0.3, fill_color=BLACK,
            fill_opacity=0.78, stroke_width=0).move_to([0, -0.1, 0])
        indep_lines.move_to(indep_bg)
        indep_cap = VGroup(indep_bg, indep_lines)
        self.play(Create(indep_edges), FadeIn(indep_cap), run_time=0.8)
        self.wait(4.0)
        self.play(FadeIn(indep_crosses), run_time=0.4)
        self.play(FadeOut(indep_edges), FadeOut(indep_crosses),
                  FadeOut(indep_cap), run_time=0.6)
        self.wait(1.3)

        # ── Inference: evidence flows UP the arrows and reshapes the belief ──
        # Detected objects are present; the rest of the vocabulary is absent.
        # Both kinds of evidence count (Bernoulli naive Bayes). No on-screen
        # caption here — this beat is narrated (see scenariusz).
        present = detected
        absent = extra_objects
        for slot in range(len(detected)):
            self.play(graph.mark_present(slot), run_time=0.30)
        for slot in range(len(detected), len(all_objects)):
            self.play(graph.mark_absent(slot), run_time=0.30)
        self.wait(0.5)

        dots, flow_anim = graph.pulse(range(len(all_objects)), color=YELLOW)
        self.add(dots)
        self.play(flow_anim, run_time=1.0)
        # remove the group AND each dot: play() registers the animated dots in
        # the scene individually, so removing only the VGroup leaves them behind.
        # self.play(FadeOut(dots), *[FadeOut(d) for d in dots])

        posterior = naive_bayes_posterior(ROOMS, ROOM_OBJECT_WEIGHT,
                                          present, absent)
        win = int(np.argmax(posterior))
        self.play(
            graph.set_posterior(posterior, highlight=win, color=GREEN),
            FadeOut(dots), *[FadeOut(d) for d in dots],
            run_time=1.0,
        )
        self.wait(1.0)

        win_caption = _banner(f"Najbardziej prawdopodobne: {ROOMS[win]}",
                              GREEN, font_size=30)
        self.play(FadeIn(win_caption), run_time=0.6)
        self.wait(1.5)
        # Everything (graph + leftover dots already removed) clears before the
        # next scene (s04) opens on the formula.
        self.play(FadeOut(VGroup(graph, win_caption)))
