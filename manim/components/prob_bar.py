import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme


class ProbabilityBars(VGroup):
    """Horizontal stacked probability bars (one per room)."""

    def __init__(self, items, max_width=4.5, bar_h=0.42, gap=0.18,
                 label_width=2.2, colors=None, font_size=22, **kwargs):
        """
        items : list of (label_str, probability_float)
        """
        super().__init__(**kwargs)
        if colors is None:
            palette = theme.PROB_BAR_PALETTE
            colors = [palette[i % len(palette)] for i in range(len(items))]

        self.bars = []
        for idx, (label, prob) in enumerate(items):
            y = -idx * (bar_h + gap)

            lbl = Text(label, font_size=font_size).set_width(label_width)
            lbl.move_to(LEFT * (max_width / 2 + label_width / 2 + 0.25) + UP * y)

            bg = Rectangle(width=max_width, height=bar_h,
                           fill_color=DARK_GRAY, fill_opacity=0.5, stroke_width=0)
            bg.move_to(UP * y)

            fill_w = max(max_width * prob, 0.01)
            fill = Rectangle(width=fill_w, height=bar_h,
                             fill_color=colors[idx], fill_opacity=0.9, stroke_width=0)
            fill.align_to(bg, LEFT)

            pct = Text(f"{prob * 100:.1f}%", font_size=font_size - 2)
            pct.next_to(bg, RIGHT, buff=0.2)
            pct.align_to(bg, UP)

            row = VGroup(lbl, bg, fill, pct)
            self.bars.append(row)
            self.add(row)

    def animate_in(self, scene, lag=0.12):
        for row in self.bars:
            lbl, bg, fill, pct = row
            scene.play(
                FadeIn(lbl, shift=RIGHT * 0.2),
                FadeIn(bg),
                GrowFromEdge(fill, LEFT),
                FadeIn(pct),
                run_time=0.6,
            )
