"""
Scene 1a – Kitchen photo with YOLO boxes → 'Kuchnia' label
Scene 1b – Robot with camera → pixel grid → building map
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import theme
import numpy as np

IMG_DIR = os.path.join(os.path.dirname(__file__), "../assets/images")


class SceneIntroRoom(Scene):
    def construct(self):
        # self._enter_title()
        # ── kitchen photo ──────────────────────────────────────────────────
        img = ImageMobject(os.path.join(IMG_DIR, "kitchen.jpg")).set_height(7.0)
        img.move_to(ORIGIN)
        self.play(FadeIn(img, run_time=1.2))
        self.wait(1.4)

        # ── bounding boxes ─────────────────────────────────────────────────
        # Coords from YOLOv8m (conf > 0.79) on kitchen.jpg (5472×3648 px).
        # Converted to Manim units: scale=7.0/3648, img half-width=5.25.
        # fmt: (label, color, [x, y, 0], box_w, box_h)
        boxes_data = [
            ("lodówka",      GREEN,  [ 1.79, -0.58, 0], 2.22, 4.24),  # conf 0.95
            # ("mikrofalówka", BLUE,   [-1.38,  0.99, 0], 1.31, 0.86),  # conf 0.92
            ("zlew",         YELLOW, [-4.10, -0.73, 0], 2.30, 0.54),  # conf 0.80
            ("piekarnik",    ORANGE, [-1.63, -0.85, 0], 1.75, 2.20),  # conf 0.91
        ]
        from components.bbox import BoundingBox
        boxes = []
        for label, color, pos, bw, bh in boxes_data:
            box = BoundingBox(width=bw, height=bh, label=label, color=color)
            box.move_to(np.array(pos))
            boxes.append(box)

        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.8, run_time=2.4))
        self.wait(0.5)

        # ── "Kuchnia" central label ────────────────────────────────────────
        room_label = Text("Kuchnia", font_size=80, color=WHITE, weight=BOLD)
        room_label.move_to(ORIGIN)

        label_bg = RoundedRectangle(
            corner_radius=0.2,
            width=room_label.width + 0.6,
            height=room_label.height + 0.4,
            fill_color=BLACK, fill_opacity=0.75, stroke_width=0,
        ).move_to(ORIGIN)

        self.play(FadeIn(label_bg), Write(room_label), run_time=0.9)
        self.wait(0.3)

        # ── boxes fly into the label ───────────────────────────────────────
        self.play(
            *[b.animate.move_to(ORIGIN).scale(0.05).set_opacity(0) for b in boxes],
            run_time=1.0,
        )
        self.play(
            room_label.animate.scale(1.15),
            rate_func=there_and_back, run_time=0.5,
        )
        self.wait(1.0)
        self.play(FadeOut(img), FadeOut(label_bg), FadeOut(room_label))


    def _enter_title(self):
        title = theme.crisp_text(
            "Sieć Bayesa w semantycznym\nrozpoznawaniu pomieszczeń",
            font_size=56, color=theme.TITLE, weight=BOLD, line_spacing=1.0,
        )
        title.center()
        self.play(Write(title), run_time=2.4)
        self.wait(1.8)
        self.play(FadeOut(title, shift=UP * 0.5), run_time=0.7)


class SceneIntroRobot(Scene):
    def construct(self):
        # ── robot silhouette ───────────────────────────────────────────────
        body  = Rectangle(width=1.1, height=1.5, color=theme.ROBOT_STROKE,
                          fill_color=theme.ROBOT_FILL, fill_opacity=0.4).shift(LEFT * 3.5)
        head  = Rectangle(width=0.9, height=0.7, color=theme.ROBOT_STROKE,
                          fill_color=theme.ROBOT_FILL, fill_opacity=0.4)
        head.next_to(body, UP, buff=0.07)
        eye_l = Dot(radius=0.10, color=theme.ACCENT_LIGHT).move_to(head.get_center() + LEFT * 0.18 + UP * 0.05)
        eye_r = Dot(radius=0.10, color=theme.ACCENT_LIGHT).move_to(head.get_center() + RIGHT * 0.18 + UP * 0.05)
        leg_l = Line(body.get_bottom() + LEFT * 0.25, body.get_bottom() + LEFT * 0.25 + DOWN * 0.6,
                     color=theme.ROBOT_STROKE, stroke_width=5)
        leg_r = Line(body.get_bottom() + RIGHT * 0.25, body.get_bottom() + RIGHT * 0.25 + DOWN * 0.6,
                     color=theme.ROBOT_STROKE, stroke_width=5)
        # camera on top-right of head
        cam = Rectangle(width=0.35, height=0.25, color=YELLOW,
                        fill_color=YELLOW, fill_opacity=0.6)
        cam.next_to(head, RIGHT, buff=0.05).align_to(head, UP)
        cam_lens = Circle(radius=0.09, color=BLACK, fill_color=BLACK, fill_opacity=0.8)
        cam_lens.move_to(cam)

        robot = VGroup(body, head, eye_l, eye_r, leg_l, leg_r, cam, cam_lens)
        robot.center()

        self.play(FadeIn(robot, shift=UP * 0.3), run_time=0.9)
        self.wait(1.3)

        # ── pixel / number stream from camera ─────────────────────────────
        pixel_vals = ["128", "043", "255", "017", "200", "089", "167", "034",
                      "211", "076", "145", "098", "003", "187", "122", "061"]
        pixel_group = VGroup()
        cols, rows = 4, 4
        for r in range(rows):
            for c in range(cols):
                val = pixel_vals[r * cols + c]
                cell_bg = Square(side_length=0.42,
                                 fill_color=interpolate_color(BLACK, theme.ACCENT, int(val) / 255),
                                 fill_opacity=0.7, stroke_color=GRAY_C, stroke_width=0.8)
                cell_txt = Text(val, font_size=52, color=WHITE).scale(0.25)
                cell_txt.move_to(cell_bg)
                pixel_group.add(VGroup(cell_bg, cell_txt))

        pixel_group.arrange_in_grid(rows, cols, buff=0.04)
        pixel_group.next_to(cam, RIGHT, buff=0.6).align_to(cam, UP)
        pixel_group.shift(DOWN * 0.1)

        beam = Line(cam.get_right(), pixel_group.get_left(), color=YELLOW_A, stroke_width=1.5)

        self.play(Create(beam), FadeIn(pixel_group, shift=RIGHT * 0.3), run_time=0.8)
        self.wait(2.5)

        # ── question text ──────────────────────────────────────────────────
        q = Text("Gdzie jest kuchnia?", font_size=36, color=theme.TITLE)
        q.to_edge(UP, buff=0.35)
        self.play(Write(q), run_time=0.9)
        self.wait(0.5)

        # ── simple building floor-plan ─────────────────────────────────────
        # Layout: two columns of rooms (2.0 × 1.5 each) separated by a
        # 0.4-wide corridor. Rooms share walls — no overlaps.
        #   total: (2.0 + 0.4 + 2.0) × (1.5 + 1.5)  =  4.4 × 3.0
        #
        #  ┌────────────┬──┬────────────┐
        #  │  kuchnia   │  │   salon    │
        #  ├────────────┤  ├────────────┤
        #  │ sypialnia  │  │  łazienka  │
        #  └────────────┴──┴────────────┘
        plan_rooms = [
            ("kuchnia",   "#2D6A4F", [-1.5,  1.0, 0], 2.5, 2.0),
            ("salon",     "#1B4F72", [ 1.5,  1.0, 0], 2.5, 2.0),
            ("sypialnia", "#4A235A", [-1.5, -1.0, 0], 2.5, 2.0),
            ("łazienka",  "#6E2F1A", [ 1.5, -1.0, 0], 2.5, 2.0),
            ("korytarz",  "#424949", [ 0.0,  0.0, 0], 0.5, 4.0),
        ]

        plan_group = VGroup()
        highlight_rect = None
        for name, color, pos, w, h in plan_rooms:
            r = Rectangle(width=w, height=h, color=WHITE,
                          fill_color=color, fill_opacity=0.35, stroke_width=1.5)
            r.move_to(pos)
            lbl = Text(name, font_size=80, color=WHITE).scale(0.2)
            lbl.move_to(r)
            if name == "korytarz":
                lbl.rotate(PI / 2)
            plan_group.add(VGroup(r, lbl))
            if name == "kuchnia":
                highlight_rect = r

        outer = Rectangle(width=5.5, height=4.0, color=WHITE, stroke_width=2)
        plan_group.add(outer)
        plan_group.shift(RIGHT * 2.8 + DOWN * 0.3)

        self.play(
            FadeIn(plan_group, shift=RIGHT * 0.4),
            robot.animate.shift(LEFT * 4.5),
            pixel_group.animate.shift(LEFT * 4.5),
            beam.animate.shift(LEFT * 4.5),
            run_time=1.0,
        )

        # highlight kitchen
        self.play(
            highlight_rect.animate.set_fill(GREEN, opacity=0.75),
            run_time=0.7,
        )
        arrow_plan = Arrow(
            pixel_group.get_right(), highlight_rect.get_left(), buff=0.15,
            color=GREEN, stroke_width=2.5,
        )
        self.play(Create(arrow_plan), run_time=0.5)
        self.wait(1.2)
        self.play(FadeOut(VGroup(robot, beam, pixel_group, q, plan_group, arrow_plan)))
