from manim import *


class BoundingBox(VGroup):
    """Labeled detection box (like a YOLO output overlay)."""

    def __init__(self, width=2.0, height=1.5, label="object",
                 confidence=None, color=GREEN, font_size=22, **kwargs):
        super().__init__(**kwargs)
        rect = Rectangle(width=width, height=height, color=color, stroke_width=3)

        tag_parts = [label]
        if confidence is not None:
            tag_parts.append(f"{confidence}%")
        tag_text = "  ".join(tag_parts)

        tag_label = Text(tag_text, font_size=font_size, color=BLACK, weight=BOLD)
        tag_bg = Rectangle(
            width=tag_label.width + 0.2,
            height=tag_label.height + 0.15,
            color=color,
            fill_color=color,
            fill_opacity=0.85,
            stroke_width=0,
        )
        tag_bg.align_to(rect, UL).shift(UP * 0.01)
        tag_label.move_to(tag_bg)

        self.rect = rect
        self.tag = VGroup(tag_bg, tag_label)
        self.add(rect, self.tag)
