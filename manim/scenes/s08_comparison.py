"""
Scene 8 – Side-by-side comparison: our model vs the BORM paper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme
from components.pipeline import PipelineBlock, build_pipeline

IMG_DIR = os.path.join(os.path.dirname(__file__), "../assets/images")


class SceneComparison(Scene):
    def construct(self):
        # ── intro: source article ──────────────────────────────────────────
        src_title = Text("Artykuł źródłowy", font_size=40, color=theme.TITLE, weight=BOLD)
        src_title.to_edge(UP, buff=0.5)
        article = ImageMobject(os.path.join(IMG_DIR, "article.png")).set_height(5.0)
        article.next_to(src_title, DOWN, buff=0.4)

        self.play(Write(src_title))
        self.play(FadeIn(article, shift=UP * 0.2))
        self.wait(1.5)
        self.play(FadeOut(article, shift=DOWN * 0.2), FadeOut(src_title))

        # ── transition to comparison ───────────────────────────────────────
        title = Text("Nasz model vs artykuł BORM", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ── our pipeline (left) ────────────────────────────────────────────
        our_specs = [
            ("Zdjęcie",           "",                     GRAY_B),
            ("YOLO",              "nazwy obiektów",       theme.ACCENT),
            ("Naiwny Bayes",      "star-sieć",            theme.HIGHLIGHT),
            ("Pomieszczenie",     "",                     GREEN_C),
        ]
        our_blocks, our_arrows = build_pipeline(our_specs, spacing=0.45, vertical=True)
        our_group = VGroup(*our_blocks, *our_arrows)
        our_group.center().shift(LEFT * 3.5 + DOWN * 0.3)

        our_label = Text("NASZ MODEL", font_size=88, color=theme.TITLE, weight=BOLD).scale(0.25)
        our_label.next_to(our_group, UP, buff=0.3)

        # ── paper pipeline (right) ─────────────────────────────────────────
        art_specs = [
            ("Zdjęcie",           "",                     GRAY_B),
            ("CNN",               "wektor cech",          PURPLE),
            ("Sieć decyzyjna",    "zależności",           RED_C),
            ("Pomieszczenie",     "",                     GREEN_C),
        ]
        art_blocks, art_arrows = build_pipeline(art_specs, spacing=0.45, vertical=True)
        art_group = VGroup(*art_blocks, *art_arrows)
        art_group.center().shift(RIGHT * 3.5 + DOWN * 0.3)

        art_label = Text("ARTYKUŁ", font_size=88, color=PURPLE, weight=BOLD).scale(0.25)
        art_label.next_to(art_group, UP, buff=0.3)

        divider = DashedLine(UP * 3.5, DOWN * 3.5, color=GRAY_D, stroke_width=1.5)

        # animate
        self.play(FadeIn(divider))
        self.play(
            LaggedStart(
                FadeIn(our_label), FadeIn(our_blocks[0]),
                lag_ratio=0.3, run_time=0.6,
            ),
            LaggedStart(
                FadeIn(art_label), FadeIn(art_blocks[0]),
                lag_ratio=0.3, run_time=0.6,
            ),
        )
        for i in range(len(our_arrows)):
            self.play(
                GrowArrow(our_arrows[i]), FadeIn(our_blocks[i + 1]),
                GrowArrow(art_arrows[i]), FadeIn(art_blocks[i + 1]),
                run_time=0.55,
            )
            self.wait(0.1)

        self.wait(0.8)

        # ── annotations (replace the pipelines) ────────────────────────────
        ann_our = VGroup(
            Text("✓ proste, interpretowalny", font_size=68, color=GREEN_A).scale(0.25),
            Text("✓ szybkie trenowanie",       font_size=68, color=GREEN_A).scale(0.25),
            Text("✗ zakłada niezależność cech",font_size=68, color=RED_A).scale(0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        ann_our.move_to(our_group)

        ann_art = VGroup(
            Text("✓ uchwytuje zależności cech", font_size=68, color=GREEN_A).scale(0.25),
            Text("✓ bogatsza reprezentacja",     font_size=68, color=GREEN_A).scale(0.25),
            Text("✗ wymaga więcej danych",       font_size=68, color=RED_A).scale(0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        ann_art.move_to(art_group)

        # fade the graphs out while the pros/cons fade in
        self.play(
            FadeOut(our_group), FadeOut(art_group),
            FadeIn(ann_our, shift=UP * 0.1), FadeIn(ann_art, shift=UP * 0.1),
        )
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, our_label, art_label,
                                  ann_our, ann_art, divider)))
