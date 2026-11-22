"""Build the dataset and artifacts on a fresh clone if they are missing, so `pytest` always runs."""
import subprocess
import sys
from pathlib import Path

import pytest

from src import config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts():
    if not config.RAW_PATH.exists():
        subprocess.run([sys.executable, "-m", "src.generate_data"], cwd=ROOT, check=True)
    if not (config.MODEL_PATH.exists() and config.PIPELINE_PATH.exists() and config.REFERENCE_STATS_PATH.exists()):
        subprocess.run([sys.executable, "-m", "src.train"], cwd=ROOT, check=True)
