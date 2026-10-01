from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Paths
MODEL_PATH = ARTIFACTS_DIR / "model" / "pipeline.pkl"
META_PATH = ARTIFACTS_DIR / "model" / "metadata.json"