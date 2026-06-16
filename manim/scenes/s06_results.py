"""
Scene 6a – Confusion matrix heatmap (drawn in Manim, real values from animation_data.json)
Scene 6b – Two visually similar rooms (corridor vs staircase)
Scene 6c – Per-room metrics (precision / recall / F1 grouped bars)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme
import json, numpy as np

DATA    = json.load(open(os.path.join(os.path.dirname(__file__), "../data/animation_data.json")))
IMG_DIR = os.path.join(os.path.dirname(__file__), "../assets/images")

ROOMS = DATA["rooms"]
CM    = np.array(DATA["confusion_matrix"], dtype=float)

ROOM_PL = {
    "office": "biuro", "bedroom": "sypialnia", "kitchen": "kuchnia",
    "bathroom": "łazienka",
    "livingroom": "salon", "living_room": "salon",
    "dining_room": "jadalnia",
    "corridor": "korytarz",
    "library": "biblioteka", "library-indoor": "biblioteka",
    "gym": "siłownia",     "gymnasium-indoor": "siłownia",
    "garage": "garaż",     "garage-indoor": "garaż",
    "staircase": "schody"
}


class SceneResults(Scene):
    def construct(self):
        self._confusion_matrix()
        self.wait(0.5)
        self._similar_rooms()
        self.wait(0.5)
        self._per_room_metrics()

    # ── Confusion matrix ───────────────────────────────────────────────────
    def _confusion_matrix(self):
        title = Text("Macierz pomyłek", font_size=36, color=theme.TITLE)
        title.to_edge(UP, buff=0.1)
        self.play(Write(title))

        n = len(ROOMS)
        row_sums = CM.sum(axis=1, keepdims=True)
        cm_norm  = CM / np.maximum(row_sums, 1)   # row-normalised

        cell = 0.52

        # grid — centered on screen, nudged down to leave room for title
        cells = VGroup()
        cell_rects = {}
        for i in range(n):
            for j in range(n):
                v = float(cm_norm[i, j])
                sq = Square(
                    side_length=cell,
                    fill_color=theme.HIGHLIGHT,
                    fill_opacity=min(v, 1.0),
                    stroke_color=GRAY_D, stroke_width=0.5,
                )
                sq.move_to(RIGHT * j * cell + DOWN * i * cell)
                cells.add(sq)
                cell_rects[(i, j)] = sq

        cells.center().shift(DOWN * 0.3)

        labels_pl = [ROOM_PL.get(r, r) for r in ROOMS]

        # row labels (true) — positioned after cells are placed, so matrix doesn't move
        row_lbls = VGroup()
        for i, name in enumerate(labels_pl):
            t = Text(name, font_size=48, color=GRAY_A).scale(0.25)
            t.next_to(cell_rects[(i, 0)], LEFT, buff=0.12)
            row_lbls.add(t)

        # column labels (predicted)
        col_lbls = VGroup()
        for j, name in enumerate(labels_pl):
            t = Text(name, font_size=48, color=GRAY_A).scale(0.25)
            t.rotate(PI / 2)
            t.next_to(cell_rects[(0, j)], UP, buff=0.12)
            col_lbls.add(t)

        axis_true = Text("prawdziwa", font_size=64, color=GRAY_B).scale(0.25)
        axis_true.next_to(row_lbls, LEFT, buff=0.18).rotate(PI / 2)
        axis_pred = Text("przewidywana", font_size=64, color=GRAY_B).scale(0.25)
        axis_pred.next_to(col_lbls, UP, buff=0.18)

        # cell value labels (only where raw count > 0, mirroring evaluation plot)
        cell_vals = VGroup()
        for i in range(n):
            for j in range(n):
                v = float(cm_norm[i, j])
                if int(CM[i, j]) == 0:
                    continue
                txt_color = WHITE if v > 0.5 else GRAY_B
                lbl = Text(f"{v:.2f}", font_size=52, color=txt_color).scale(0.25)
                lbl.move_to(cell_rects[(i, j)])
                cell_vals.add(lbl)

        self.play(FadeIn(cells, shift=UP * 0.15), run_time=1.0)
        self.play(FadeIn(cell_vals), run_time=0.6)
        self.play(FadeIn(row_lbls), FadeIn(col_lbls), run_time=0.5)
        self.play(FadeIn(axis_true), FadeIn(axis_pred), FadeOut(title))
        self.wait(0.5)

        # highlight diagonal
        diag = [cell_rects[(i, i)] for i in range(n)]
        self.play(
            *[c.animate.set_stroke(color=WHITE, width=2.5) for c in diag],
            run_time=0.7,
        )
        diag_lbl = Text("Przekątna = poprawne klasyfikacje",
                        font_size=80, color=WHITE).scale(0.25)
        diag_lbl.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(diag_lbl, shift=UP * 0.1))
        self.wait(1.0)

        # highlight worst room (lowest recall) — transition text in place
        per_room_m = DATA["metrics"]["per_room"]
        recalls = [per_room_m[r]["recall"] for r in ROOMS]
        worst_i  = int(np.argmin(recalls))
        worst_name = ROOM_PL.get(ROOMS[worst_i], ROOMS[worst_i])
        worst_recall_pct = int(round(recalls[worst_i] * 100))

        worst_row = [cell_rects[(worst_i, j)] for j in range(n) if j != worst_i]
        self.play(
            *[c.animate.set_stroke(color=RED, width=2.0) for c in worst_row],
            run_time=0.6,
        )
        err_lbl = Text(
            f"{worst_name}: recall = {worst_recall_pct} % (pomylona z innymi!)",
            font_size=80, color=RED).scale(0.25)
        err_lbl.move_to(diag_lbl)
        self.play(ReplacementTransform(diag_lbl, err_lbl))
        self.wait(1.2)
        self.play(FadeOut(VGroup(cells, cell_vals,
                                 row_lbls, col_lbls, axis_true, axis_pred, err_lbl)))

    # ── Similar rooms ──────────────────────────────────────────────────────
    def _similar_rooms(self):
        title = Text("Problem — podobne wizualnie pomieszczenia",
                     font_size=30, color=ORANGE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        img_cor = ImageMobject(os.path.join(IMG_DIR, "corridor.jpg")).set_height(5.5)
        img_sta = ImageMobject(os.path.join(IMG_DIR, "staircase.jpg")).set_height(5.5)
        img_cor.shift(LEFT * 3.2)
        img_sta.shift(RIGHT * 3.2)

        lbl_cor = Text("Korytarz", font_size=28, color=GRAY_A)
        lbl_sta = Text("Klatka schodowa", font_size=28, color=GRAY_A)
        lbl_cor.next_to(img_cor, DOWN, buff=0.2)
        lbl_sta.next_to(img_sta, DOWN, buff=0.2)

        self.play(FadeIn(img_cor), FadeIn(img_sta), run_time=0.8)
        self.play(FadeIn(lbl_cor), FadeIn(lbl_sta))
        self.wait(0.5)

        q_cor = Text("?", font_size=72, color=RED)
        q_sta = Text("?", font_size=72, color=RED)
        q_cor.move_to(img_cor)
        q_sta.move_to(img_sta)

        self.play(FadeIn(q_cor, scale=1.5), FadeIn(q_sta, scale=1.5))
        self.wait(0.4)

        note = theme.crisp_text(
            "Brak charakterystycznych obiektów",
            font_size=24, color=GRAY_A,
        )
        note.to_edge(DOWN, buff=0.30)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.5)
        self.play(FadeOut(Group(title, img_cor, img_sta,
                                 lbl_cor, lbl_sta, q_cor, q_sta, note)))

    # ── Per-room metrics ───────────────────────────────────────────────────
    def _per_room_metrics(self):
        title = Text("Metryki dla pomieszczeń", font_size=34, color=theme.TITLE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        per_room  = DATA["metrics"]["per_room"]
        labels_pl = [ROOM_PL.get(r, r) for r in ROOMS]
        series    = [("precyzja", "precision", BLUE_C),
                     ("czułość",  "recall",    GREEN_C),
                     ("F1",       "f1",        GOLD_C)]

        # ── plot geometry (a value of 1.0 spans `scale_h` units) ──────────
        n        = len(ROOMS)
        baseline = -1.9
        scale_h  = 3.4
        left     = -5.9
        group_w  = (2 * abs(left)) / n          # width allotted per room
        bar_w    = group_w / 4.0                # 3 bars + padding

        # y-axis with gridlines + tick labels
        axis = VGroup()
        y_axis = Line([left - 0.15, baseline, 0], [left - 0.15, baseline + scale_h, 0],
                      stroke_color=GRAY_C, stroke_width=2)
        x_axis = Line([left - 0.15, baseline, 0], [-left + 0.15, baseline, 0],
                      stroke_color=GRAY_C, stroke_width=2)
        axis.add(x_axis, y_axis)
        grid = VGroup()
        for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
            y = baseline + frac * scale_h
            grid.add(DashedLine([left - 0.15, y, 0], [-left + 0.15, y, 0],
                                stroke_color=theme.DIVIDER, stroke_width=0.8,
                                dash_length=0.06).set_opacity(0.45))
            tick = Text(f"{frac:.1f}", font_size=48, color=GRAY_B).scale(0.25)
            tick.next_to([left - 0.15, y, 0], LEFT, buff=0.12)
            grid.add(tick)

        self.play(Create(axis), FadeIn(grid), run_time=0.7)

        # ── bars + room labels ───────────────────────────────────────────
        bars      = VGroup()
        room_lbls = VGroup()
        for i, room in enumerate(ROOMS):
            cx = left + group_w * (i + 0.5)
            for s, (_, key, color) in enumerate(series):
                v = float(per_room[room][key])
                x = cx + (s - 1) * bar_w
                bar = Rectangle(
                    width=bar_w * 0.92, height=max(v * scale_h, 1e-3),
                    fill_color=color, fill_opacity=0.9, stroke_width=0,
                )
                bar.move_to([x, baseline + v * scale_h / 2, 0])
                bars.add(bar)
            lbl = Text(labels_pl[i], font_size=44, color=GRAY_A).scale(0.25)
            lbl.rotate(25 * DEGREES)
            lbl.next_to([cx, baseline, 0], DOWN, buff=0.12)
            room_lbls.add(lbl)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.02),
            run_time=1.6,
        )
        self.play(FadeIn(room_lbls), run_time=0.5)

        # ── legend ───────────────────────────────────────────────────────
        legend = VGroup()
        for name, _, color in series:
            swatch = Square(side_length=0.22, fill_color=color,
                            fill_opacity=0.9, stroke_width=0)
            txt = Text(name, font_size=52, color=GRAY_A).scale(0.25)
            txt.next_to(swatch, RIGHT, buff=0.12)
            legend.add(VGroup(swatch, txt))
        legend.arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(legend), run_time=0.5)
        self.wait(1.0)

        # callout: two worst rooms by F1
        f1s = [(ROOM_PL.get(r, r), per_room[r]["f1"]) for r in ROOMS]
        f1s.sort(key=lambda x: x[1])

        self.wait(1.5)
        self.play(FadeOut(VGroup(title, axis, grid, bars, room_lbls, legend)))
