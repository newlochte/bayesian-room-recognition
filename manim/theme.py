from manim import *

Text.set_default(font="Noto Sans", disable_ligatures=True)


def crisp_text(text, font_size=24, _oversample=8, **kwargs):
    """Text with even letter spacing.

    Pango kerns/hints poorly at small font sizes, giving uneven gaps between
    letters. Rendering at a large size and scaling the vector down preserves
    the good spacing while landing at the requested visual size.
    """
    return Text(text, font_size=font_size * _oversample, **kwargs).scale(1 / _oversample)


def crisp_text_arrow(text, font_size=24, color=WHITE, arrow=r"\rightarrow",
                     buff=0.18, line_buff=0.22, arrow_scale=0.62, **kwargs):
    """Like crisp_text, but every '→' (or '->') becomes a real LaTeX arrow.

    Pango text handles the Polish glyphs; the arrow is a MathTex so it renders
    as a proper typographic arrow instead of the font's stick glyph. Supports
    multiple arrows per line and '\\n' newlines. Returns a VGroup that can be
    positioned/animated like any text mobject.
    """
    import re
    rows = []
    for line in text.split("\n"):
        parts = re.split(r"\s*(?:→|->)\s*", line)
        row = VGroup()
        ref_h = None
        for k, seg in enumerate(parts):
            if k > 0:
                row.add(MathTex(arrow, color=color))   # sized after we know ref_h
            seg = seg.strip()
            if seg:
                t = crisp_text(seg, font_size=font_size, color=color, **kwargs)
                ref_h = ref_h or t.height
                row.add(t)
        ref_h = ref_h or 0.3
        for m in row:
            if isinstance(m, MathTex):
                m.scale_to_fit_height(ref_h * arrow_scale)
        row.arrange(RIGHT, buff=buff)
        rows.append(row)
    return rows[0] if len(rows) == 1 else VGroup(*rows).arrange(DOWN, buff=line_buff)

# ── Teal-green palette ─────────────────────────────────────────────────────────
# Change these values to retheme every scene at once.

TITLE          = GREEN_B   # scene / section titles
ACCENT         = GREEN_C   # pipeline blocks, structural nodes
ACCENT_LIGHT   = GREEN_A   # light annotations, links
ACCENT_DARK    = GREEN_E   # dark body fills (e.g. robot)
HIGHLIGHT      = GREEN     # Bayesian graph center, matrix fill
HIGHLIGHT_LITE = GREEN_A   # Bayesian row labels, secondary highlights

POSITIVE       = GREEN_A  # ✓ items
NEGATIVE       = RED_A    # ✗ items
MUTED          = GRAY_A   # secondary / caption text
EDGE           = GRAY_C   # graph edges, thin lines
DIVIDER        = GRAY_D   # dashed separators

# Robot silhouette
ROBOT_STROKE   = GREEN_C
ROBOT_FILL     = GREEN_D

# Bayesian graph component defaults
BAYES_CENTER   = GREEN
BAYES_LEAF     = GREEN_C

# Probability bar default palette
PROB_BAR_PALETTE = [GREEN_B, GREEN_C, GREEN_D, GREEN_E, GREEN_A, GREEN]
