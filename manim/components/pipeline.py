import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme


class PipelineBlock(VGroup):
    def __init__(self, title, subtitle="", color=None,
                 width=3.2, height=1.1, font_size=23, **kwargs):
        super().__init__(**kwargs)
        if color is None:
            color = theme.ACCENT
        rect = RoundedRectangle(
            corner_radius=0.18, width=width, height=height,
            color=color, fill_color=color, fill_opacity=0.18, stroke_width=2.5,
        )
        title_text = Text(title, font_size=font_size, color=WHITE, weight=BOLD)
        title_text.move_to(rect)
        if subtitle:
            sub = Text(subtitle, font_size=(font_size - 6) * 4, color=GRAY_A).scale(0.25)
            sub.next_to(title_text, DOWN, buff=0.08)
            title_text.shift(UP * 0.12)
            self.add(rect, title_text, sub)
        else:
            self.add(rect, title_text)
        self.rect = rect


def build_pipeline(specs, spacing=0.7, vertical=False):
    """
    specs : list of (title, subtitle, color)
    Returns (blocks, arrows) – both are lists.
    """
    blocks, arrows = [], []
    direction = DOWN if vertical else RIGHT

    for i, (title, subtitle, color) in enumerate(specs):
        blk = PipelineBlock(title, subtitle=subtitle, color=color)
        if i > 0:
            blk.next_to(blocks[-1], direction, buff=spacing)
        blocks.append(blk)

    for i in range(len(blocks) - 1):
        if vertical:
            start = blocks[i].get_bottom()
            end   = blocks[i + 1].get_top()
        else:
            start = blocks[i].get_right()
            end   = blocks[i + 1].get_left()
        arr = Arrow(start, end, buff=0.08, color=GRAY_A, stroke_width=2.5)
        arrows.append(arr)

    return blocks, arrows
