"""
Render individual scenes or all of them.

Quick preview (low quality, opens in viewer):
    cd manim/
    ../.venv/bin/manim -pql scenes/s01_intro.py SceneIntroRoom

High quality render of one scene:
    ../.venv/bin/manim -pqh scenes/s04_bayesian.py SceneBayes

Render everything (see render_all.sh):
    bash render_all.sh

Scene → class mapping
─────────────────────────────────────────────────────────────────
s01_intro.py       SceneIntroRoom      intro: kitchen photo + boxes
                   SceneIntroRobot     intro: robot + floor plan
s02_architecture.py SceneArchitecture  two-block pipeline overview
s03_yolo.py        SceneYolo           YOLO detection + label list
s04_bayesian.py    SceneBayes          star graph + formula + edge weights
s05_training.py    SceneTraining       prob table + Laplace formula
s06_results.py     SceneResults        confusion matrix + similar rooms
s07_demo.py        SceneDemo           3 live demos (kitchen/bedroom/corridor)
s08_comparison.py  SceneComparison     our model vs BORM paper
s09_summary.py     SceneSummary        per-room F1 bars + YOLO limitation
s10_credits.py     SceneCredits        bibliography + "Dziękuję"
─────────────────────────────────────────────────────────────────
"""
