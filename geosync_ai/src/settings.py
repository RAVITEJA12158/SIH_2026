"""Project-local paths used by the runnable prototype.

Keeping these paths relative to the repository makes `python src/pipeline.py`
work on a developer machine, rather than depending on a previous environment.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_project_directories():
    """Create the local generated-artifact directories when needed."""
    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
