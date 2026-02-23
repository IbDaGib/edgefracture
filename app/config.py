"""Shared configuration constants for the EdgeFracture app."""

import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MEDGEMMA_MODEL = os.environ.get(
    "MEDGEMMA_MODEL",
    "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M",
)

SAFETY_AUDIT_ENABLED = (
    os.environ.get("SAFETY_AUDIT_ENABLED", "true").lower() == "true"
)
SAFETY_AUDIT_TIMEOUT = int(os.environ.get("SAFETY_AUDIT_TIMEOUT", "180"))

TRIAGE_THRESHOLDS = {"red": 0.70, "yellow": 0.40}
VALIDATED_REGIONS = {"Hand", "Leg", "Hip", "Shoulder"}
CXR_SOFT_LOCK_MARGIN = float(os.environ.get("CXR_SOFT_LOCK_MARGIN", "0.05"))

LINEAR_PROBE_PATH = os.environ.get(
    "PROBE_PATH",
    "results/linear_probe/fracture_probe.joblib",
)
CXR_MODEL_PATH = os.environ.get(
    "CXR_MODEL_PATH",
    "models/cxr-foundation/elixr-c-v2-pooled",
)

IMAGE_SIZE = 1024  # CXR Foundation input size
