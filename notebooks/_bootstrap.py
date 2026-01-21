# notebooks/_bootstrap.py
from __future__ import annotations

import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    """
    Walk upwards from 'start' until a directory containing 'src/' is found.
    """
    cur = start.resolve()
    for _ in range(10):  # safety limit
        if (cur / "src").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError("Could not find project root containing 'src' directory.")

# This file lives in notebooks/, so start from there
_NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(_NOTEBOOK_DIR)

# Make sure project root is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Standard output directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Default seeds for experiments
DEFAULT_SEEDS = list(range(10))

print(f"[bootstrap] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[bootstrap] RESULTS_DIR  = {RESULTS_DIR}")
print(f"[bootstrap] FIGURES_DIR  = {FIGURES_DIR}")