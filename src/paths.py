"""
paths.py - single source of truth for every folder/file CartIQ reads or writes.

All paths are built from this file's own location (not the current working
directory), so scripts behave the same no matter where they are launched from
(PyCharm run button, `uv run`, a different terminal folder, ...).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"            # raw Instacart CSVs (read-only, never modified)
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PROCESSED_DIR = OUTPUTS_DIR / "processed"   # small aggregate tables + ML feature inputs (parquet)

MODEL_PATH = MODELS_DIR / "cartiq_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
