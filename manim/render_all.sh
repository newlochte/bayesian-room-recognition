#!/usr/bin/env bash
# Render all manim scenes at high quality.
# Run from the manim/ directory:
#   cd manim && bash render_all.sh
#
# Low-quality preview mode: bash render_all.sh -ql
# Flag is forwarded to every manim call.

MANIM=manim
QUALITY=${1:--qp}   # default: high quality

set -e
echo "=== Rendering all scenes (quality flag: $QUALITY) ==="

render() {
    local file=$1
    local scene=$2
    echo -e "\n--- $file :: $scene ---"
    $MANIM $QUALITY "$file" "$scene"
}

render scenes/s01_intro.py        SceneIntroRoom
render scenes/s01_intro.py        SceneIntroRobot
render scenes/s02_architecture.py SceneArchitecture
render scenes/s03_yolo.py         SceneYoloBayes
render scenes/s04_bayesian.py     SceneBayes
render scenes/s05_training.py     SceneTraining
render scenes/s06_results.py      SceneResults
render scenes/s07_demo.py         SceneDemo
render scenes/s08_comparison.py   SceneComparison
render scenes/s09_summary.py      SceneSummary
render scenes/s10_credits.py      SceneCredits

echo -e "\n=== Done. Videos are in media/videos/ ==="
