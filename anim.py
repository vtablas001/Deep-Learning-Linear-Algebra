"""
Deep Learning Linear Algebra Animation
====================================================
A visual companion to the Deep-Learning-Linear-Algebra README.
WHAT these operations mean and WHY they matter.
"""

from manim import *
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#0f0f1a"
BLUE_A   = "#4fc3f7"
ORANGE_A = "#ffb74d"
GREEN_A  = "#81c784"
PINK_A   = "#f48fb1"
PURPLE_A = "#ce93d8"
RED_A2   = "#ef5350"
SOFT_W   = "#e0e0e0"

config.background_color = BG


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 1 — Title
# ═══════════════════════════════════════════════════════════════════════════════
class S01_Title(Scene):
    def construct(self):
        title = Text(
            "Deep Learning\nLinear Algebra",
            font_size=58, color=WHITE, line_spacing=1.2,
        ).shift(UP * 0.5)
        sub = Text(
            "The math your neural network executes on every forward pass",
            font_size=21, color=BLUE_A, slant=ITALIC,
        ).next_to(title, DOWN, buff=0.6)
        author = Text("Victor A. Tablas", font_size=18, color=GRAY_B
                       ).next_to(sub, DOWN, buff=0.8)

        # ambient dots
        dots = VGroup()
        rng = np.random.default_rng(42)
        for _ in range(35):
            x, y = rng.uniform(-7, 7), rng.uniform(-4, 4)
            dots.add(Dot([x, y, 0], radius=0.03, color=BLUE_A).set_opacity(0.15))

        self.play(FadeIn(dots, lag_ratio=0.01), run_time=0.8)
        self.play(Write(title), run_time=2)
        self.play(FadeIn(sub, shift=UP * 0.15), run_time=1)
        self.play(FadeIn(author, shift=UP * 0.15), run_time=0.7)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 2 — The Image-to-Matrix Bridge  (README §1)
# ═══════════════════════════════════════════════════════════════════════════════
class S02_ImageToMatrix(Scene):
    def construct(self):
        hdr = Text("1. The Image-to-Matrix Bridge", font_size=36, color=BLUE_A)
        self.play(Write(hdr)); self.wait(0.4)
        self.play(hdr.animate.scale(0.55).to_edge(UP, buff=0.25))

        # — Pixel grid representing an image —
        grid_size = 6
        pixel_grid = VGroup()
        rng = np.random.default_rng(7)
        for r in range(grid_size):
            for c in range(grid_size):
                v = rng.uniform(0.15, 0.95)
                sq = Square(side_length=0.42, stroke_width=0.5, stroke_color=GRAY_C,
                            fill_color=interpolate_color(BLACK, WHITE, v),
                            fill_opacity=1)
                sq.move_to([c * 0.44 - 1.1, -r * 0.44 + 1.1, 0])
                pixel_grid.add(sq)
        pixel_grid.shift(LEFT * 3.5)

        img_label = Text("One 6x6 grayscale image", font_size=16, color=GRAY_B
                         ).next_to(pixel_grid, DOWN, buff=0.25)
        self.play(FadeIn(pixel_grid, lag_ratio=0.008), FadeIn(img_label), run_time=1.2)
        self.wait(1)

        # — 4-D tensor notation —
        tensor_tex = MathTex(
            r"\mathcal{X}", r"\in", r"\mathbb{R}^{N \times H \times W \times C}",
            font_size=36,
        ).shift(RIGHT * 2 + UP * 1.8)
        tensor_tex[0].set_color(ORANGE_A)

        parts = VGroup(
            Text("N = batch of images", font_size=16, color=GREEN_A),
            Text("H x W = height x width", font_size=16, color=PINK_A),
            Text("C = color channels", font_size=16, color=PURPLE_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(tensor_tex, DOWN, buff=0.35)

        self.play(Write(tensor_tex), run_time=1)
        self.play(FadeIn(parts, lag_ratio=0.3), run_time=1.2)
        self.wait(1.5)

        # — Flatten arrow —
        arrow = Arrow(LEFT * 0.8, RIGHT * 0.8, color=ORANGE_A, stroke_width=3).shift(DOWN * 0.3)
        flatten_label = Text("flatten", font_size=18, color=ORANGE_A
                             ).next_to(arrow, UP, buff=0.1)

        # — Flat vector —
        flat_vec = Matrix(
            [["p_1", "p_2", "\\dots", "p_{36}"]],
            left_bracket="[", right_bracket="]",
            element_to_mobject_config={"font_size": 22},
        ).scale(0.75).shift(RIGHT * 2 + DOWN * 0.3)

        flat_eq = MathTex(
            r"D = H \times W \times C = 36", font_size=22, color=SOFT_W,
        ).next_to(flat_vec, DOWN, buff=0.3)

        self.play(GrowArrow(arrow), FadeIn(flatten_label), run_time=0.8)
        self.play(Write(flat_vec), run_time=1)
        self.play(FadeIn(flat_eq), run_time=0.6)
        self.wait(1)

        # — Insight —
        insight = Text(
            "Before a network can learn from a photo,\n"
            "the grid of pixels must become a flat row of numbers.\n"
            "That row IS the vector x your equations operate on.",
            font_size=17, color=BLUE_A, line_spacing=1.4,
        ).shift(DOWN * 2.5)
        self.play(FadeIn(insight, shift=UP * 0.15), run_time=1.2)
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 3 — Weights, Bias & Broadcasting  (README §2-3)
# ═══════════════════════════════════════════════════════════════════════════════
class S03_WeightsBiasBroadcast(Scene):
    def construct(self):
        hdr = Text("2-3. Weights, Bias & Broadcasting", font_size=34, color=BLUE_A)
        self.play(Write(hdr)); self.wait(0.3)
        self.play(hdr.animate.scale(0.55).to_edge(UP, buff=0.25))

        # — Weight matrix —
        w_label = Text("Weight Matrix W", font_size=20, color=PINK_A).shift(LEFT * 4 + UP * 2)
        w_mat = Matrix(
            [["w_{11}", "w_{12}"],
             ["w_{21}", "w_{22}"],
             ["w_{31}", "w_{32}"]],
            left_bracket="[", right_bracket="]",
            element_to_mobject_config={"font_size": 22},
        ).scale(0.7).next_to(w_label, DOWN, buff=0.2)
        w_dim = MathTex(r"D \times M", font_size=20, color=GRAY_B
                        ).next_to(w_mat, DOWN, buff=0.15)
        w_meaning = Text(
            "Each column is one neuron's\n\"recipe\" for combining inputs.",
            font_size=14, color=GRAY_C, line_spacing=1.3,
        ).next_to(w_dim, DOWN, buff=0.2)

        self.play(Write(w_label), Write(w_mat), FadeIn(w_dim), run_time=1.2)
        self.play(FadeIn(w_meaning, shift=UP * 0.1), run_time=0.8)

        # — Bias vector —
        b_label = Text("Bias Vector b", font_size=20, color=BLUE_A).shift(RIGHT * 1 + UP * 2)
        b_vec = Matrix(
            [["b_1", "b_2"]],
            left_bracket="[", right_bracket="]",
            element_to_mobject_config={"font_size": 22},
        ).scale(0.7).next_to(b_label, DOWN, buff=0.2)
        b_dim = MathTex(r"1 \times M", font_size=20, color=GRAY_B
                        ).next_to(b_vec, DOWN, buff=0.15)
        b_meaning = Text(
            "A baseline offset.\nWithout it, the neuron's\noutput must pass through zero.",
            font_size=14, color=GRAY_C, line_spacing=1.3,
        ).next_to(b_dim, DOWN, buff=0.2)

        self.play(Write(b_label), Write(b_vec), FadeIn(b_dim), run_time=1)
        self.play(FadeIn(b_meaning, shift=UP * 0.1), run_time=0.8)
        self.wait(1.5)

        # — Broadcasting —
        bc_title = Text("Broadcasting the bias across the batch",
                        font_size=20, color=ORANGE_A).shift(DOWN * 0.2)
        bc_eq = MathTex(
            r"\mathbf{B} = \mathbf{1}_N \, b = "
            r"\begin{bmatrix}1\\1\\1\end{bmatrix}"
            r"\begin{bmatrix}b_1 & b_2\end{bmatrix}"
            r"= \begin{bmatrix}b_1 & b_2\\b_1 & b_2\\b_1 & b_2\end{bmatrix}",
            font_size=24,
        ).next_to(bc_title, DOWN, buff=0.25)

        bc_insight = Text(
            "Every sample in the batch gets the same bias.\n"
            "The ones-vector is how we write that \"copy-paste\" in math.",
            font_size=16, color=SOFT_W, line_spacing=1.3,
        ).next_to(bc_eq, DOWN, buff=0.35)

        self.play(Write(bc_title), run_time=0.8)
        self.play(Write(bc_eq), run_time=2)
        self.play(FadeIn(bc_insight, shift=UP * 0.1), run_time=1)
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 4 — The Linear Transformation  (README §4)
# ═══════════════════════════════════════════════════════════════════════════════
class S04_LinearTransform(Scene):
    def construct(self):
        hdr = Text("4. The Linear Transformation", font_size=36, color=BLUE_A)
        self.play(Write(hdr)); self.wait(0.3)
        self.play(hdr.animate.scale(0.55).to_edge(UP, buff=0.25))

        # — The equation —
        eq = MathTex(
            r"Y", r"=", r"X", r"W", r"+", r"\mathbf{1}_N", r"b",
            font_size=52,
        ).shift(UP * 1.5)
        eq[0].set_color(ORANGE_A)   # Y
        eq[2].set_color(GREEN_A)    # X
        eq[3].set_color(PINK_A)     # W
        eq[5].set_color(GRAY_B)     # 1_N
        eq[6].set_color(BLUE_A)     # b

        self.play(Write(eq), run_time=2)
        self.wait(1)

        # — Dimension check —
        dim_title = Text("Dimension Consistency Check", font_size=18, color=ORANGE_A
                         ).shift(UP * 0.3)
        dims = VGroup(
            MathTex(r"X W : \;(N \!\times\! D)\cdot(D \!\times\! M) = (N \!\times\! M)",
                    font_size=24, color=GREEN_A),
            MathTex(r"\mathbf{1}_N b : \;(N \!\times\! 1)\cdot(1 \!\times\! M) = (N \!\times\! M)",
                    font_size=24, color=BLUE_A),
            MathTex(r"Y : \;(N \!\times\! M)",
                    font_size=24, color=ORANGE_A),
        ).arrange(DOWN, buff=0.25).next_to(dim_title, DOWN, buff=0.25)

        self.play(FadeIn(dim_title), run_time=0.5)
        for d in dims:
            self.play(Write(d), run_time=0.8)
            self.wait(0.4)
        self.wait(1)

        # — Geometric insight (mini transformation) —
        insight_box = RoundedRectangle(
            width=10, height=1.6, corner_radius=0.15,
            stroke_color=BLUE_A, stroke_width=1,
            fill_color=BG, fill_opacity=0.95,
        ).shift(DOWN * 2.5)
        insight_text = Text(
            "This single line of math is the entire forward pass\n"
            "of a fully connected layer. Multiply to mix features,\n"
            "add bias to shift the decision boundary.",
            font_size=17, color=SOFT_W, line_spacing=1.35,
        ).move_to(insight_box)

        self.play(FadeIn(insight_box), Write(insight_text), run_time=1.5)
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 5 — Two-Layer Network & ReLU  (README §5)
# ═══════════════════════════════════════════════════════════════════════════════
class S05_TwoLayerNet(Scene):
    def construct(self):
        hdr = Text("5. The Two-Layer Neural Network", font_size=34, color=BLUE_A)
        self.play(Write(hdr)); self.wait(0.3)
        self.play(hdr.animate.scale(0.55).to_edge(UP, buff=0.25))

        # — Step 1: Z1 —
        s1_label = Text("Step 1: Hidden linear transform", font_size=18, color=GREEN_A
                        ).shift(UP * 2 + LEFT * 2.5)
        s1_eq = MathTex(r"Z_1 = X W_1 + \mathbf{1}_N b_1", font_size=32
                        ).next_to(s1_label, DOWN, buff=0.2)
        self.play(Write(s1_label), Write(s1_eq), run_time=1.2)

        # — Step 2: ReLU —
        s2_label = Text("Step 2: Activation (ReLU)", font_size=18, color=ORANGE_A
                        ).next_to(s1_eq, DOWN, buff=0.5)
        s2_eq = MathTex(r"A_1 = \max(0,\; Z_1)", font_size=32
                        ).next_to(s2_label, DOWN, buff=0.2)
        self.play(Write(s2_label), Write(s2_eq), run_time=1.2)

        # — Mini ReLU graph —
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-0.5, 3, 1],
            x_length=3.5, y_length=2,
            axis_config={"stroke_color": GRAY_C, "stroke_width": 1.5},
        ).shift(RIGHT * 4 + UP * 0.2)
        relu_graph = axes.plot(lambda x: max(0, x), color=ORANGE_A, use_smoothing=False)
        relu_lbl = Text("ReLU", font_size=14, color=ORANGE_A).next_to(axes, UP, buff=0.1)
        relu_meaning = Text(
            "\"If negative, silence it.\nIf positive, let it through.\"",
            font_size=13, color=GRAY_B, line_spacing=1.3,
        ).next_to(axes, DOWN, buff=0.15)

        self.play(Create(axes), Create(relu_graph), FadeIn(relu_lbl), run_time=1.2)
        self.play(FadeIn(relu_meaning), run_time=0.8)
        self.wait(1)

        # — Step 3: Output —
        s3_label = Text("Step 3: Output linear transform", font_size=18, color=PINK_A
                        ).next_to(s2_eq, DOWN, buff=0.5)
        s3_eq = MathTex(r"Y = A_1 W_2 + \mathbf{1}_N b_2", font_size=32
                        ).next_to(s3_label, DOWN, buff=0.2)
        self.play(Write(s3_label), Write(s3_eq), run_time=1.2)
        self.wait(1)

        # — Full chained equation —
        chain_box = RoundedRectangle(
            width=10, height=1.2, corner_radius=0.12,
            stroke_color=PURPLE_A, stroke_width=1.5,
            fill_color=BG, fill_opacity=0.95,
        ).shift(DOWN * 2.8)
        chain_eq = MathTex(
            r"Y = \sigma(X W_1 + \mathbf{1}_N b_1)\, W_2 + \mathbf{1}_N b_2",
            font_size=30, color=PURPLE_A,
        ).move_to(chain_box)

        self.play(FadeIn(chain_box), Write(chain_eq), run_time=1.5)
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 6 — Backpropagation  (README §7, §10)
# ═══════════════════════════════════════════════════════════════════════════════
class S06_Backprop(Scene):
    def construct(self):
        hdr = Text("7-10. Backpropagation", font_size=36, color=BLUE_A)
        self.play(Write(hdr)); self.wait(0.3)
        self.play(hdr.animate.scale(0.55).to_edge(UP, buff=0.25))

        question = Text(
            "How does the network learn which weights to change?",
            font_size=22, color=ORANGE_A,
        ).next_to(hdr, DOWN, buff=0.3)
        self.play(Write(question), run_time=1.2)

        # — Chain of boxes representing layers —
        box_data = [
            ("X", GREEN_A),
            ("Z\u2081 = XW\u2081+b\u2081", GRAY_B),
            ("A\u2081 = ReLU", ORANGE_A),
            ("Z\u2082 = A\u2081W\u2082+b\u2082", GRAY_B),
            ("Loss L", RED_A2),
        ]
        boxes = VGroup()
        for label_text, clr in box_data:
            box = VGroup(
                RoundedRectangle(width=2.2, height=0.8, corner_radius=0.1,
                                 stroke_color=clr, stroke_width=1.5,
                                 fill_color=BG, fill_opacity=0.9),
                Text(label_text, font_size=14, color=clr),
            )
            box[1].move_to(box[0])
            boxes.add(box)
        boxes.arrange(RIGHT, buff=0.3).shift(UP * 0.5)

        # Forward arrows
        fwd_arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i][0].get_right(), boxes[i + 1][0].get_left(),
                buff=0.05, color=GREEN_A, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
            fwd_arrows.add(a)

        fwd_label = Text("Forward Pass", font_size=16, color=GREEN_A
                         ).next_to(fwd_arrows, UP, buff=0.15)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in boxes],
                              lag_ratio=0.15), run_time=1.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in fwd_arrows],
                              lag_ratio=0.1), FadeIn(fwd_label), run_time=1)
        self.wait(1.5)

        # Backward arrows
        bwd_arrows = VGroup()
        for i in range(len(boxes) - 1, 0, -1):
            a = Arrow(
                boxes[i][0].get_left() + DOWN * 0.5,
                boxes[i - 1][0].get_right() + DOWN * 0.5,
                buff=0.05, color=RED_A2, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
            bwd_arrows.add(a)

        bwd_label = Text("Backward Pass (gradients)", font_size=16, color=RED_A2
                         ).next_to(bwd_arrows, DOWN, buff=0.15)
        self.play(LaggedStart(*[GrowArrow(a) for a in bwd_arrows],
                              lag_ratio=0.15), FadeIn(bwd_label), run_time=1.5)
        self.wait(1)

        # — Gradient equations —
        grad_title = Text("The three gradients at each layer:", font_size=18, color=SOFT_W
                          ).shift(DOWN * 1.5)
        grads = VGroup(
            MathTex(r"dW = X^T \cdot dY", font_size=26, color=PINK_A),
            MathTex(r"db = \mathbf{1}_N^T \cdot dY", font_size=26, color=BLUE_A),
            MathTex(r"dX = dY \cdot W^T", font_size=26, color=GREEN_A),
        ).arrange(RIGHT, buff=0.8).next_to(grad_title, DOWN, buff=0.25)

        grad_labels = VGroup(
            Text("update the weights", font_size=12, color=PINK_A),
            Text("update the bias", font_size=12, color=BLUE_A),
            Text("pass error backward", font_size=12, color=GREEN_A),
        )
        for gl, g in zip(grad_labels, grads):
            gl.next_to(g, DOWN, buff=0.12)

        self.play(FadeIn(grad_title), run_time=0.5)
        self.play(LaggedStart(*[Write(g) for g in grads], lag_ratio=0.2), run_time=1.5)
        self.play(FadeIn(grad_labels, lag_ratio=0.2), run_time=0.8)
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 7 — Adam Optimizer  (README §8.3)
# ═══════════════════════════════════════════════════════════════════════════════
class S07_AdamOptimizer(Scene):
    def construct(self):
        hdr = Text("8. Adam: The Optimizer", font_size=36, color=BLUE_A)
        self.play(Write(hdr)); self.wait(0.3)
        self.play(hdr.animate.scale(0.55).to_edge(UP, buff=0.25))

        # — Economist analogy —
        analogy = Text(
            "\"As an economist, I view optimizers as systems\n"
            "balancing historical trends (momentum)\n"
            "with market volatility (adaptive rates).\"",
            font_size=18, color=ORANGE_A, line_spacing=1.35,
            slant=ITALIC,
        ).shift(UP * 2)
        self.play(FadeIn(analogy, shift=UP * 0.15), run_time=1.5)
        self.wait(2)
        self.play(analogy.animate.scale(0.7).to_edge(UP, buff=0.6).set_opacity(0.4))

        # — Step 1: Moments —
        s1 = Text("Step 1: Track trend + volatility", font_size=20, color=GREEN_A
                   ).shift(UP * 1.2 + LEFT * 3)
        m_eq = MathTex(
            r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\, dW",
            font_size=26,
        ).next_to(s1, DOWN, buff=0.2)
        m_lbl = Text("trend (momentum)", font_size=14, color=GREEN_A
                     ).next_to(m_eq, RIGHT, buff=0.3)
        v_eq = MathTex(
            r"v_t = \beta_2 v_{t-1} + (1-\beta_2)\, dW^2",
            font_size=26,
        ).next_to(m_eq, DOWN, buff=0.25)
        v_lbl = Text("volatility (variance)", font_size=14, color=PINK_A
                     ).next_to(v_eq, RIGHT, buff=0.3)

        self.play(Write(s1), run_time=0.6)
        self.play(Write(m_eq), FadeIn(m_lbl), run_time=1)
        self.play(Write(v_eq), FadeIn(v_lbl), run_time=1)
        self.wait(1)

        # — Step 2: Bias correction —
        s2 = Text("Step 2: Bias correction", font_size=20, color=ORANGE_A
                   ).next_to(v_eq, DOWN, buff=0.5)
        bc_eqs = MathTex(
            r"\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad "
            r"\hat{v}_t = \frac{v_t}{1-\beta_2^t}",
            font_size=26,
        ).next_to(s2, DOWN, buff=0.2)
        bc_why = Text(
            "Early iterations are biased toward zero.\n"
            "This scales them up, then gracefully fades away.",
            font_size=14, color=GRAY_B, line_spacing=1.3,
        ).next_to(bc_eqs, DOWN, buff=0.2)

        self.play(Write(s2), run_time=0.6)
        self.play(Write(bc_eqs), run_time=1.2)
        self.play(FadeIn(bc_why, shift=UP * 0.1), run_time=0.8)
        self.wait(1.5)

        # — Step 3: Update —
        update_box = RoundedRectangle(
            width=8, height=1.3, corner_radius=0.12,
            stroke_color=PURPLE_A, stroke_width=1.5,
            fill_color=BG, fill_opacity=0.95,
        ).shift(DOWN * 2.8)
        update_eq = MathTex(
            r"W_t = W_{t-1} - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}",
            font_size=30, color=PURPLE_A,
        ).move_to(update_box)
        update_meaning = Text(
            "Step in the direction of the trend, but slow down for volatile weights.",
            font_size=15, color=SOFT_W,
        ).next_to(update_box, DOWN, buff=0.15)

        self.play(FadeIn(update_box), Write(update_eq), run_time=1.5)
        self.play(FadeIn(update_meaning), run_time=0.8)
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE 8 — Full Lifecycle Summary  (README §9)
# ═══════════════════════════════════════════════════════════════════════════════
class S08_Lifecycle(Scene):
    def construct(self):
        hdr = Text("9. The Full Lifecycle", font_size=36, color=BLUE_A)
        self.play(Write(hdr)); self.wait(0.3)
        self.play(hdr.animate.scale(0.55).to_edge(UP, buff=0.25))

        # — Three-phase cycle —
        phases = [
            ("FORWARD", GREEN_A,
             "Push X through layers\nto get predictions Y"),
            ("LOSS", RED_A2,
             "Measure how wrong\nthe prediction was"),
            ("BACKWARD", ORANGE_A,
             "Compute gradients:\nhow to fix each weight"),
            ("UPDATE", PURPLE_A,
             "Adam adjusts W and b\nusing trend + volatility"),
        ]

        # Arrange in a cycle
        positions = [UP * 1.5, RIGHT * 4, DOWN * 1.5, LEFT * 4]
        phase_groups = VGroup()
        for (label, clr, desc), pos in zip(phases, positions):
            box = RoundedRectangle(
                width=2.8, height=1.6, corner_radius=0.12,
                stroke_color=clr, stroke_width=2,
                fill_color=BG, fill_opacity=0.95,
            ).move_to(pos)
            title = Text(label, font_size=18, color=clr, weight=BOLD
                         ).move_to(box.get_top() + DOWN * 0.3)
            body = Text(desc, font_size=13, color=SOFT_W, line_spacing=1.3
                        ).move_to(box.get_center() + DOWN * 0.15)
            phase_groups.add(VGroup(box, title, body))

        # Arrows between phases
        cycle_arrows = VGroup()
        for i in range(4):
            start_box = phase_groups[i][0]
            end_box = phase_groups[(i + 1) % 4][0]
            a = Arrow(
                start_box.get_edge_center(
                    RIGHT if i == 0 else DOWN if i == 1 else LEFT if i == 2 else UP
                ),
                end_box.get_edge_center(
                    LEFT if i == 0 else UP if i == 1 else RIGHT if i == 2 else DOWN
                ),
                buff=0.1, color=GRAY_B, stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            cycle_arrows.add(a)

        # Center label
        center_text = Text("repeat until\nconvergence", font_size=16,
                           color=GRAY_C, line_spacing=1.3).move_to(ORIGIN)

        self.play(
            LaggedStart(*[FadeIn(pg, scale=0.9) for pg in phase_groups], lag_ratio=0.2),
            run_time=2,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in cycle_arrows], lag_ratio=0.15),
            FadeIn(center_text),
            run_time=1.5,
        )
        self.wait(3)

        # — Closing insight —
        closing = Text(
            "Every AI model you have ever used\n"
            "runs this exact loop millions of times.\n"
            "Linear algebra is the engine.",
            font_size=20, color=BLUE_A, line_spacing=1.4,
        ).shift(DOWN * 3.2)
        self.play(FadeIn(closing, shift=UP * 0.2), run_time=1.5)
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1)
