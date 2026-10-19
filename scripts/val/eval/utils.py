"""Compatibility entry point for the shared offline/validation math grader."""
from pathlib import Path
import sys

# Support standalone `python scripts/val/eval/grade.py` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verl"))
from verl.utils.reward_score.math_evaluation import *  # noqa: F401,F403
