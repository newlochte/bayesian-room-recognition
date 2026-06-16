"""
Scene 2 – Pipeline overview:  [Zdjęcie] → [YOLO] → [Sieć bayesowska]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme
from components.pipeline import PipelineBlock, build_pipeline


class SceneArchitecture(Scene):
    def construct(self):
        title = Text("Jak działa system?", font_size=40, color=theme.TITLE)
        title.to_edge(UP, buff=0.2)
        self.play(Write(title), run_time=0.8)
        self.wait(3.0)

        # (title, subtitle, color, wait_after)
        specs = [
            ("Zdjęcie",          "",GRAY_B,          1.0),
            ("Detekcja obiektów","",theme.POSITIVE,    0.75),
            ("Klasyfikacja",     "",theme.ACCENT, 0.75),
            ("Pomieszczenie",    "",BLUE_C,  0.5),
        ]
        blocks, arrows = build_pipeline(
            [(t, s, c) for t, s, c, _ in specs], spacing=0.55, vertical=True
        )

        group = VGroup(*blocks, *arrows)
        group.center().shift(DOWN * 0.15)

        # animate blocks + arrows appearing sequentially
        self.play(FadeIn(blocks[0]), run_time=0.5)
        self.wait(specs[0][3])
        for i in range(len(arrows)):
            self.play(
                GrowArrow(arrows[i]),
                FadeIn(blocks[i + 1]),
                run_time=0.6,
            )
            self.wait(specs[i + 1][3])

        self.wait(1.5)
        self.play(FadeOut(VGroup(title, group)))
