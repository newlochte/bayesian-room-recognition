"""
Scene 10 – Bibliography, animation credit, authors, then fade out.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme


class SceneCredits(Scene):
    def construct(self):
        self._bibliography()
        self._animacje()
        self._authors()
        self._thanks()

    # ── Screen 1: bibliography / sources ────────────────────────────────────
    def _bibliography(self):
        lines = [
            ("Artykuł:", theme.TITLE, 30),
            ('L. Zhou et al., "BORM: Bayesian Object Relation Model', GRAY_A, 22),
            ('for Indoor Scene Recognition," IROS 2021.', GRAY_A, 22),
            ("arXiv:2108.00397", theme.ACCENT_LIGHT, 22),
            ("", WHITE, 10),
            ("Dataset:", theme.TITLE, 30),
            ("Places365 (MIT CSAIL)", GRAY_A, 22),
            ("", WHITE, 10),
            ("Detekcja obiektów:", theme.TITLE, 30),
            ("YOLOv8m  (Ultralytics)", GRAY_A, 22),
            ("", WHITE, 10),
            ("Repozytorium:", theme.TITLE, 30),
            ("github.com/newlochte/bayesian-room-recognition", theme.ACCENT_LIGHT, 22),
        ]

        text_mobs = VGroup()
        reveal = []   # only the visible mobs get an entrance animation
        for text, color, size in lines:
            if text == "":
                # invisible strut: adds vertical gap between sections
                text_mobs.add(Rectangle(width=0.01, height=size / 100,
                                        stroke_opacity=0, fill_opacity=0))
            else:
                mob = theme.crisp_text(text, font_size=size, color=color)
                text_mobs.add(mob)
                reveal.append(mob)
        text_mobs.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        text_mobs.center().shift(UP * 0.2)

        for mob in reveal:
            self.play(FadeIn(mob, shift=RIGHT * 0.15), run_time=0.35)
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in text_mobs], run_time=0.8)

    # ── Screen 2: animation credit — Manim logo on its own screen ───────────
    def _animacje(self):
        label = theme.crisp_text("Animacje", font_size=30, color=theme.TITLE)
        label.to_edge(UP, buff=1.2)
        banner = ManimBanner().scale(0.5)

        self.play(FadeIn(label))
        self.play(banner.create())
        self.play(banner.expand())
        self.wait(1.5)
        self.play(FadeOut(banner), FadeOut(label))

    # ── Screen 3: authors (one name per line) + low acknowledgements ────────
    def _authors(self):
        names = VGroup(
            theme.crisp_text("Wykonanie", font_size=30, color=theme.TITLE),
            theme.crisp_text("Karolina Michalak", font_size=30, color=WHITE),
            theme.crisp_text("Tymoteusz Tomczak", font_size=30, color=WHITE),
        ).arrange(DOWN, buff=0.28)
        names.move_to(UP * 0.6)

        sub = VGroup(
            theme.crisp_text("Projekt wykonany na zajęcia ze \nSztucznej Inteligencji w Robotyce",
                             font_size=20, color=GRAY_A),
            theme.crisp_text("Podziękowania dla pani Joanny Piasek-Skupnej",
                             font_size=20, color=theme.ACCENT_LIGHT),
        ).arrange(DOWN, buff=0.18)
        sub.to_edge(DOWN, buff=0.6)

        self.play(FadeIn(names, shift=UP * 0.1), run_time=0.8)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(names), FadeOut(sub), run_time=0.8)

    # ── Final beat ──────────────────────────────────────────────────────────
    def _thanks(self):
        thanks = theme.crisp_text("Dziękujemy", font_size=72, color=WHITE)
        self.play(Write(thanks), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(thanks, run_time=1.2))
