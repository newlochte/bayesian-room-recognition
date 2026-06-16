"""
Scene 4 – Naive Bayes formula, then presence/absence as evidence.

Split out from s03 (SceneYoloBayes): s03 reveals the star network and runs the
"which room?" inference; s04 opens on the formula that justifies it, then shows
how presence AND absence of objects move the posterior on the same star graph.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from manim import *
import theme
from components.bayes_star import BayesStarGraph, naive_bayes_posterior
# Single source of truth for the room hypotheses, abstract P(obj|room) weights
# and the centered-banner helper — defined alongside the s03 scene.
from s03_yolo import ROOMS, ROOM_OBJECT_WEIGHT, _banner


class SceneBayes(Scene):
    def construct(self):
        self._scene_formula()
        self.wait(0.5)
        self._scene_edge_weights()
        self.wait(0.5)
        self._scene_formula_absence()

    @staticmethod
    def _basic_formula(font_size=40):
        return MathTex(
            r"P(\text{room} \mid \text{objects})",
            r"\;\propto\;",
            r"P(\text{room})",
            r"\cdot",
            r"\prod_{i}",
            r"P(\text{obj}_i \mid \text{room})",
            font_size=font_size,
        )

    @staticmethod
    def _absence_formula(font_size=32):
        return MathTex(
            r"P(\text{room} \mid \text{objects})",
            r"\;\propto\;",
            r"P(\text{room})",
            r"\cdot",
            r"\prod_{\text{present}} P(\text{obj} \mid \text{room})",
            r"\cdot",
            r"\prod_{\text{absent}} (1 - P(\text{obj} \mid \text{room}))",
            font_size=font_size,
        )

    # ── Part 1: basic formula + the open question (first thing in s04) ─────
    def _scene_formula(self):
        title = theme.crisp_text("Założenie Bayesa", font_size=32,
                                 color=theme.HIGHLIGHT)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(1.0)

        formula = self._basic_formula().center()
        self.play(Write(formula), run_time=1.5)
        self.wait(1.0)

        prior = formula[2]
        self.play(
            Indicate(prior, color=theme.HIGHLIGHT, scale_factor=1.5),
            Flash(prior, color=theme.HIGHLIGHT, flash_radius=0.55,
                  line_length=0.22, num_lines=10),
            run_time=0.8,
        )
        self.wait(0.5)

        likelihood = formula[5]
        self.play(
            Indicate(likelihood, color=theme.HIGHLIGHT, scale_factor=1.5),
            Flash(likelihood, color=theme.HIGHLIGHT, flash_radius=0.75,
                  line_length=0.22, num_lines=10),
            run_time=0.8,
        )
        self.wait(2.6)

        note = theme.crisp_text(
            "Czy brak obiektu może być dowodem przeciw?",
            font_size=23, color=GRAY_A,
        )
        note.next_to(formula, DOWN, buff=0.6)
        self.play(FadeIn(note, shift=UP * 0.2))
        self.wait(1.2)
        # Leave on the open question — the graph answers it; we come back after.
        self.play(FadeOut(VGroup(title, formula, note)))

    # ── Part 2: presence / absence as evidence (same star network) ─────────
    def _scene_edge_weights(self):
        objects_subset = ["łóżko", "monitor", "klawiatura", "kuchenka", "kanapa", "zlew"]
        graph = BayesStarGraph(ROOMS, objects_subset, object_width=10.5,
                               object_r=0.36, font_size_object=16, node_y=2.15)

        self.play(FadeIn(graph.room_node, shift=DOWN * 0.25), run_time=0.8)
        self.play(
            LaggedStart(*[FadeIn(o, shift=UP * 0.25) for o in graph.objects],
                        lag_ratio=0.1),
            run_time=1.0,
        )
        self.play(LaggedStart(*[GrowArrow(a) for a in graph.arrows],
                              lag_ratio=0.05), run_time=0.9)
        self.wait(0.4)

        bed_idx     = objects_subset.index("łóżko")
        oven_idx    = objects_subset.index("kuchenka")
        monitor_idx = objects_subset.index("monitor")

        # one caption at the very bottom, swapped per step
        self._caption = None

        def step(idx, color, present, absent, text, highlight=None):
            # caption swap happens together with the object node changing
            self.play(graph.mark_present(idx) if color == GREEN
                      else graph.mark_absent(idx),
                      *self._caption_anims(text, color),
                      run_time=0.5)
            dots, flow = graph.pulse([idx], color=color)
            self.add(dots)
            self.play(flow, run_time=0.8)
            self.remove(dots, *dots)
            post = naive_bayes_posterior(ROOMS, ROOM_OBJECT_WEIGHT,
                                         present=present, absent=absent)
            self.play(graph.set_posterior(post, highlight=highlight, color=GREEN),
                      run_time=0.8)
            self.wait(0.8)
            return post

        # 1) bed absent → evidence AGAINST the bedroom
        step(bed_idx, RED, present=[], absent=["łóżko"],
             text="Brak łóżka → dowód PRZECIW sypialni")

        # 2) oven absent → evidence AGAINST the kitchen
        step(oven_idx, RED, present=[], absent=["łóżko", "kuchenka"],
             text="Brak kuchenki → dowód PRZECIW kuchni")

        # 3) monitor present → evidence FOR the office (winner highlighted)
        post = naive_bayes_posterior(ROOMS, ROOM_OBJECT_WEIGHT,
                                     present=["monitor"], absent=["łóżko", "kuchenka"])
        win = int(np.argmax(post))
        step(monitor_idx, GREEN, present=["monitor"], absent=["łóżko", "kuchenka"],
             text="Monitor obecny → dowód ZA biurem", highlight=win)

        self.wait(0.4)
        tail = [graph] + ([self._caption] if self._caption is not None else [])
        self.play(FadeOut(VGroup(*tail)))

    # fixed Y so every caption sits at the exact same height (to_edge would
    # shift the baseline with each string's glyph bounding box).
    _CAPTION_Y = -3.82

    def _caption_anims(self, text, color):
        """Return the animations to swap the single low caption, and record it.

        Returned so the caller can play them *together* with the bar update.
        """
        new = theme.crisp_text_arrow(text, font_size=26, color=color)
        new.move_to([0, self._CAPTION_Y, 0])
        anims = [FadeIn(new, shift=UP * 0.15)]
        if self._caption is not None:
            anims.append(FadeOut(self._caption))
        self._caption = new
        return anims

    # ── Part 3: back to the formula, now updated with the absence term ──────
    def _scene_formula_absence(self):
        title = theme.crisp_text("Założenie Bayesa", font_size=32,
                                 color=theme.HIGHLIGHT)
        title.to_edge(UP, buff=0.3)

        # come back to the same basic formula from Part 1...
        formula = self._basic_formula().center()
        self.play(Write(title), Write(formula), run_time=1.2)
        self.wait(1.0)

        # ...then update it with the absence product (answers the open question).
        absence_formula = self._absence_formula().center()
        self.play(Transform(formula, absence_formula), run_time=1.0)

        note = theme.crisp_text(
            "Także brak obiektu jest dowodem.",
            font_size=23, color=GRAY_A,
        )
        note.next_to(absence_formula, DOWN, buff=0.6)
        self.play(FadeIn(note, shift=UP * 0.2))
        self.wait(3.2)
        self.play(FadeOut(VGroup(title, formula, note)))
