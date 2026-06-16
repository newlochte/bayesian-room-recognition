"""
Scene 9a – Per-room F1 bar chart (real data)
Scene 9b – YOLO limitation slide
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme
import json, numpy as np

DATA    = json.load(open(os.path.join(os.path.dirname(__file__), "../data/animation_data.json")))
# current evaluation results (single source of truth for the bar chart)
EVAL    = json.load(open(os.path.join(os.path.dirname(__file__),
                                      "../../output/evaluation/metrics.json")))["val"]
IMG_DIR = os.path.join(os.path.dirname(__file__), "../assets/images")

# scene-recognition accuracy reported in the BORM paper (Zhou et al. 2021),
# Table III — reduced Places365-14 (14 classes ≈ our 11-class setup)
ART_OM   = 0.470   # object-only model (baseline) — comparable to our approach
ART_BORM = 0.749   # full Bayesian Object Relation Model

ROOM_PL = {
    "office": "biuro", "bedroom": "sypialnia", "kitchen": "kuchnia",
    "bathroom": "łazienka", "living_room": "salon", "livingroom": "salon",
    "dining_room": "jadalnia", "corridor": "korytarz",
    "library-indoor": "biblioteka", "library": "biblioteka",
    "gymnasium-indoor": "siłownia", "gym": "siłownia",
    "garage-indoor": "garaż", "garage": "garaż",
    "staircase": "schody",
}


class SceneSummary(Scene):
    def construct(self):
        self._article_comparison()
        self.wait(0.5)
        self._accuracy_bars()
        self.wait(0.5)
        self._yolo_limitation()

    # ── Article vs our model (same metric: accuracy) ───────────────────────
    def _article_comparison(self):
        title = theme.Text("Porównanie metryk dokładności",
                                 font_size=40, color=theme.TITLE)
        title.to_edge(UP, buff=0.3)
        subtitle = theme.crisp_text("Validacja na zbiorze Places365",
                                    font_size=20, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.18)
        self.play(Write(title))
        self.play(FadeIn(subtitle))

        our_acc = EVAL["accuracy"]
        entries = [
            ("Artykuł\n(OM)",   ART_OM,  PURPLE_B,        False),
            ("Nasz model",      our_acc, theme.HIGHLIGHT, True),
            ("Artykuł\n(BORM)", ART_BORM, PURPLE,         False),
        ]

        bar_max_h = 3.2
        bar_w     = 1.1
        gap       = 1.0
        n         = len(entries)

        bars_group = VGroup()
        for i, (lbl, val, color, ours) in enumerate(entries):
            x = i * (bar_w + gap) - (n - 1) * (bar_w + gap) / 2

            bar = Rectangle(
                width=bar_w, height=max(val * bar_max_h, 0.02),
                fill_color=color, fill_opacity=0.9,
                stroke_color=(YELLOW if ours else color),
                stroke_width=(3 if ours else 0),
            )
            bar.align_to(ORIGIN, DOWN).shift(RIGHT * x)

            val_txt = theme.crisp_text(f"{val:.1%}", font_size=18,
                                       color=(YELLOW if ours else WHITE))
            val_txt.next_to(bar, UP, buff=0.1)

            name_txt = theme.crisp_text(lbl, font_size=16, color=GRAY_A,
                                        line_spacing=0.6)
            name_txt.next_to(bar, DOWN, buff=0.15)

            bars_group.add(VGroup(bar, val_txt, name_txt))

        bars_group.center().shift(DOWN * 0.4)

        axis = Line(
            bars_group.get_left() + LEFT * 0.3,
            bars_group.get_right() + RIGHT * 0.3,
            color=GRAY_C, stroke_width=1.5,
        )
        axis.align_to(bars_group, DOWN)
        # keep the baseline under the bars, not their labels
        axis.set_y(bars_group[0][0].get_bottom()[1])

        self.play(FadeIn(axis))
        self.play(
            LaggedStart(
                *[GrowFromEdge(row[0], DOWN) for row in bars_group],
                lag_ratio=0.25, run_time=1.4,
            )
        )
        self.play(*[FadeIn(row[1]) for row in bars_group],
                  *[FadeIn(row[2]) for row in bars_group])
        self.wait(0.6)

        note = theme.crisp_text_arrow(
            "Nasz model dorównuje modelowi podstawowemu z artykułu.",
            font_size=20, color=YELLOW,
        )
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.8)
        note2 = theme.crisp_text_arrow(
            "Dodanie modelu relacji zwiększa dokładność.",
            font_size=20, color=YELLOW,
        )
        self.play(FadeOut(VGroup(title, subtitle, bars_group, axis, note)))

    # ── Per-room F1 bar chart ──────────────────────────────────────────────
    def _accuracy_bars(self):
        title = theme.crisp_text("Wyniki — metryka F1", font_size=30, color=theme.TITLE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        per_room = EVAL["per_room"]
        rooms    = list(per_room.keys())
        f1s      = [per_room[r]["f1"] for r in rooms]
        labels   = [ROOM_PL.get(r, r) for r in rooms]

        # Sort descending
        order  = np.argsort(f1s)[::-1]
        rooms  = [rooms[i]  for i in order]
        f1s    = [f1s[i]    for i in order]
        labels = [labels[i] for i in order]

        bar_max_h = 3.5
        bar_w     = 0.6
        gap       = 0.15
        n         = len(rooms)

        bars_group = VGroup()
        for i, (lbl, f1) in enumerate(zip(labels, f1s)):
            x = i * (bar_w + gap) - (n - 1) * (bar_w + gap) / 2

            color = interpolate_color(RED, GREEN, f1)
            bar = Rectangle(
                width=bar_w, height=max(f1 * bar_max_h, 0.02),
                fill_color=color, fill_opacity=0.85, stroke_width=0,
            )
            bar.align_to(ORIGIN, DOWN).shift(RIGHT * x)

            val_txt = theme.crisp_text(f"{f1:.2f}", font_size=14, color=WHITE)
            val_txt.next_to(bar, UP, buff=0.06)

            room_txt = theme.crisp_text(lbl, font_size=13, color=GRAY_A)
            room_txt.rotate(PI / 3)
            room_txt.next_to(bar, DOWN, buff=0.1)

            bars_group.add(VGroup(bar, val_txt, room_txt))

        bars_group.center().shift(DOWN * 0.5)

        # x-axis
        axis = Line(
            bars_group.get_left() + LEFT * 0.2,
            bars_group.get_right() + RIGHT * 0.2,
            color=GRAY_C, stroke_width=1.5,
        )
        axis.align_to(bars_group, DOWN)

        # y-axis ticks
        y_ticks = VGroup()
        for v in [0.25, 0.5, 0.75, 1.0]:
            y = v * bar_max_h
            tick = DashedLine(
                axis.get_start() + UP * y,
                axis.get_end() + UP * y,
                dash_length=0.08, color=GRAY_D, stroke_width=0.8,
            )
            tick_lbl = theme.crisp_text(f"{v:.0%}", font_size=13, color=GRAY_C)
            tick_lbl.next_to(tick, LEFT, buff=0.12)
            y_ticks.add(tick, tick_lbl)

        self.play(FadeIn(axis), FadeIn(y_ticks))
        self.play(
            LaggedStart(
                *[GrowFromEdge(row[0], DOWN) for row in bars_group],
                lag_ratio=0.08, run_time=1.5,
            )
        )
        self.play(
            *[FadeIn(row[1]) for row in bars_group],
            *[FadeIn(row[2]) for row in bars_group],
        )
        self.wait(0.6)

        # overall accuracy note
        acc = EVAL["accuracy"]
        top3 = EVAL["top3_accuracy"]
        note = theme.crisp_text(
            f"Ogólna dokładność: {acc:.1%}    Top-3: {top3:.1%}",
            font_size=24, color=YELLOW,
        )
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.2)
        self.play(FadeOut(VGroup(title, bars_group, axis, y_ticks, note)))

    # ── YOLO limitation ────────────────────────────────────────────────────
    def _yolo_limitation(self):
        title = theme.crisp_text("Ograniczenie — słownik obiektów YOLO",
                                 font_size=30, color=ORANGE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        img = ImageMobject(os.path.join(IMG_DIR, "library.jpg")).set_height(4.0)
        img.shift(LEFT * 3.0)
        self.play(FadeIn(img), run_time=0.8)

        no_det = theme.crisp_text("YOLO: (brak detekcji)", font_size=26, color=RED)
        no_det.next_to(img, DOWN, buff=0.25)
        self.play(FadeIn(no_det, shift=UP * 0.1))
        self.wait(0.4)

        explanation = VGroup(
            theme.crisp_text("Cechy niewidoczne dla YOLO:", font_size=24, color=WHITE),
            theme.crisp_text("• tapety, tekstury ścian",    font_size=22, color=GRAY_A),
            theme.crisp_text("• rodzaj podłogi",             font_size=22, color=GRAY_A),
            theme.crisp_text("• układ przestrzenny",         font_size=22, color=GRAY_A),
            theme.crisp_text("• oświetlenie",                font_size=22, color=GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        explanation.shift(RIGHT * 2.5)

        self.play(FadeIn(explanation, shift=LEFT * 0.2), run_time=0.8)
        self.wait(0.4)

        conclusion = theme.crisp_text_arrow(
            "Ograniczeniem jest reprezentacja nie model",
            font_size=21, color=YELLOW,
        )
        conclusion.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(conclusion, shift=UP * 0.1))
        self.wait(1.5)
        self.play(FadeOut(Group(title, img, no_det, explanation, conclusion)))
