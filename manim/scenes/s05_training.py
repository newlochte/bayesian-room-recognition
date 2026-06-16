"""
Scene 5 – Conditional probability table + Laplace smoothing formula.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme
import json, numpy as np

DATA = json.load(open(os.path.join(os.path.dirname(__file__), "../data/animation_data.json")))
TABLE = DATA["table"]

# friendly Polish labels
ROOM_LABELS = {
    "office": "biuro", "bedroom": "sypialnia", "kitchen": "kuchnia",
    "bathroom": "łazienka", "living_room": "salon", "corridor": "korytarz",
    "staircase": "schody"
}
OBJ_LABELS = {
    "chair": "krzesło", "bed": "łóżko", "oven": "piekarnik",
    "sink": "zlew", "couch": "kanapa", "keyboard": "klawiatura",
    "toilet": "toaleta", "dining table": "stół",
}


def _cell_color(val):
    if val >= 0.7:
        return GREEN_D
    if val >= 0.4:
        return YELLOW_D
    if val >= 0.15:
        return GOLD_D
    return DARK_GRAY


class SceneTraining(Scene):
    def construct(self):
        self._counts_to_probabilities()

    def _counts_to_probabilities(self):
        title = theme.crisp_text("Trening sieci",
                     font_size=40, color=theme.TITLE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        rooms   = [ROOM_LABELS.get(r, r) for r in TABLE["rooms"]]
        objects = [OBJ_LABELS.get(o, o) for o in TABLE["objects"]]
        values  = TABLE["values"]            # (n_rooms, n_objects) probabilities
        counts  = TABLE["counts"]            # (n_rooms, n_objects) integer counts
        n_imgs  = TABLE["images_per_room"]   # (n_rooms,) images per room

        cell_w, cell_h = 1.18, 0.48
        img_w = 1.0
        n_rows, n_cols = len(rooms), len(objects)

        # ── build the grid; each object cell starts life as an integer count ──
        all_cells = VGroup()
        row_groups = []
        num_cells = [[None] * n_cols for _ in range(n_rows)]   # (bg, num) per cell
        for i, room in enumerate(rooms):
            row_vg = VGroup()
            # row label
            lbl = Text(room, font_size=64, color=theme.HIGHLIGHT_LITE).scale(0.25)
            lbl_bg = Rectangle(width=1.45, height=cell_h,
                               fill_color=DARK_GRAY, fill_opacity=0.3, stroke_width=0.5)
            lbl.move_to(lbl_bg)
            row_vg.add(VGroup(lbl_bg, lbl))

            for j in range(n_cols):
                bg = Rectangle(width=cell_w, height=cell_h,
                               fill_color=DARK_GRAY,
                               fill_opacity=0.35, stroke_color=GRAY_D, stroke_width=0.5)
                num = Text(str(counts[i][j]), font_size=64, color=WHITE).scale(0.25)
                num.move_to(bg)
                cell = VGroup(bg, num)
                row_vg.add(cell)
                num_cells[i][j] = cell

            # trailing "images in this room" column (the denominator)
            img_bg = Rectangle(width=img_w, height=cell_h,
                               fill_color=theme.ACCENT_DARK, fill_opacity=0.45,
                               stroke_color=GRAY_D, stroke_width=0.5)
            img_num = Text(str(n_imgs[i]), font_size=64, color=theme.ACCENT_LIGHT).scale(0.25)
            img_num.move_to(img_bg)
            row_vg.add(VGroup(img_bg, img_num))

            row_vg.arrange(RIGHT, buff=0)
            all_cells.add(row_vg)
            row_groups.append(row_vg)

        all_cells.arrange(DOWN, buff=0)
        all_cells.center().shift(UP * 0.35)

        # ── column headers ───────────────────────────────────────────────────
        header_row = VGroup()
        header_row.add(Rectangle(width=1.45, height=0.36, fill_opacity=0, stroke_width=0))
        for obj in objects:
            t = Text(obj, font_size=52, color=GRAY_A).scale(0.25)
            bg = Rectangle(width=cell_w, height=0.36, fill_opacity=0, stroke_width=0)
            t.move_to(bg)
            header_row.add(VGroup(bg, t))
        img_hdr_t = Text("# obrazów", font_size=46, color=theme.ACCENT_LIGHT).scale(0.25)
        img_hdr_bg = Rectangle(width=img_w, height=0.36, fill_opacity=0, stroke_width=0)
        img_hdr_t.move_to(img_hdr_bg)
        header_row.add(VGroup(img_hdr_bg, img_hdr_t))
        header_row.arrange(RIGHT, buff=0)
        header_row.next_to(all_cells, UP, buff=0.04, aligned_edge=LEFT)
        self.wait(0.8)

        # ── Phase 1: reveal the counts ────────────────────────────────────────
        cap1 = theme.crisp_text(
            "Obliczamy prawdopodobieństwo wystąpenia",
            font_size=24, color=GRAY_A)
        cap1.next_to(all_cells, DOWN, buff=0.35)

        self.play(FadeIn(header_row, shift=DOWN * 0.1))
        for row in row_groups:
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.3)
        self.play(FadeIn(cap1, shift=UP * 0.1))
        self.wait(0.3)

        # ── Phase 2: the transition — why we can't just divide (Laplace) ──────
        zr, zc = 2, 5
        n_room = n_imgs[zr]
        alpha  = TABLE.get("smoothing", 1.0)
        p_zero = (counts[zr][zc] + alpha) / (n_room + 2 * alpha)
        zero_cell = num_cells[zr][zc]
        zero_box = SurroundingRectangle(zero_cell, color=theme.NEGATIVE, buff=0.0,
                                        stroke_width=3)

        # pre-layout sub so the equation sits at the same y-position throughout
        sub = theme.crisp_text("Wygładzanie Laplace'a",
                               font_size=30, color=theme.TITLE)
        sub.next_to(all_cells, DOWN, buff=0.3)

        # Step 1: base formula — placed below where sub will appear
        naive_formula = MathTex(
            r"P=\frac{n_{\text{obj}}}{n_{\text{pokój}}}",
            font_size=46,
        )
        naive_formula.next_to(sub, DOWN, buff=0.3)
        self.play(FadeOut(cap1, shift=DOWN * 0.1))
        self.play(Write(naive_formula), run_time=0.9)
        self.wait(0.8)

        # Step 2: highlight zero cell
        self.play(Create(zero_box))

        # Step 3: expand with numbers and warning
        naive = MathTex(
            r"P=\frac{n_{\text{obj}}}{n_{\text{pokój}}}",
            rf"=\frac{{0}}{{{n_room}}}=0",
            font_size=46,
        )
        naive.move_to(naive_formula)
        naive_warn = theme.crisp_text("✗ obiekt staje sie niemozliwy na zawsze",
                                      font_size=22, color=theme.NEGATIVE)
        naive_warn.next_to(naive, DOWN, buff=0.22)
        self.play(TransformMatchingShapes(naive_formula, naive), run_time=0.9)
        self.play(FadeIn(naive_warn, shift=UP * 0.1))
        self.wait(1.2)

        # Step 4: Wygładzanie Laplace'a label + smoothed equation together
        smoothed = MathTex(
            r"P=\frac{n_{\text{obj}}+\alpha}{n_{\text{pokój}}+2\alpha}",
            rf"=\frac{{0+1}}{{{n_room}+2}}\approx {p_zero:.3f}",
            font_size=46,
        )
        smoothed.move_to(naive)
        why = theme.crisp_text_arrow("α = 1 → żaden obiekt nie ma P = 0",
                                     font_size=22, color=theme.ACCENT_LIGHT)
        why.next_to(smoothed, DOWN, buff=0.22)
        self.play(
            TransformMatchingShapes(naive, smoothed),
            FadeIn(sub, shift=UP * 0.1),
            FadeOut(naive_warn, shift=DOWN * 0.1),
            zero_box.animate.set_stroke(color=theme.POSITIVE, width=3),
            run_time=1.2,
        )
        self.play(FadeIn(why, shift=UP * 0.1))
        self.wait(1.3)

        # ── Phase 3: reveal probabilities (counts → P, cells gain colour) ─────
        keep = MathTex(
            r"P(\text{obj}\mid\text{pokój})=\frac{n_{\text{obj}}+\alpha}{n_{\text{pokój}}+2\alpha}",
            font_size=40,
        )
        # centre it in the empty band between the table bottom and the frame edge
        band_mid_y = (all_cells.get_bottom()[1] - config.frame_y_radius) / 2
        keep.move_to([0, band_mid_y, 0])
        # morph the smoothed formula into the kept one: the general fraction
        # slides down into the empty band while the numeric example drops away.
        self.play(
            TransformMatchingShapes(smoothed, keep),
            FadeOut(why, shift=DOWN * 0.1),
            FadeOut(sub, shift=UP * 0.1),
            run_time=1.0,
        )

        morphs = []
        for i in range(n_rows):
            for j in range(n_cols):
                bg, _ = num_cells[i][j]
                target_num = Text(f"{values[i][j]:.3f}", font_size=64,
                                  color=WHITE).scale(0.25)
                target_num.move_to(bg)
                morphs.append(Transform(num_cells[i][j][1], target_num))
                morphs.append(bg.animate.set_fill(_cell_color(values[i][j]),
                                                   opacity=0.55))
        self.play(*morphs, FadeOut(zero_box), run_time=1.4)
        self.wait(0.6)

        # ── Phase 4: highlight the strongest room per object ──────────────────
        highlight_cells = []
        for j in range(n_cols):
            best_i = int(np.argmax([values[i][j] for i in range(n_rows)]))
            highlight_cells.append(num_cells[best_i][j][0])   # bg rect
        self.play(
            *[c.animate.set_stroke(color=WHITE, width=2.5) for c in highlight_cells],
            run_time=0.6,
        )
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, all_cells, header_row, keep)))
