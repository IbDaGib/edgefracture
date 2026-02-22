#!/usr/bin/env python3
"""
EdgeFracture — Portable Musculoskeletal Fracture Triage
MedGemma Impact Challenge 2026

Stepped workflow: Upload → Triage → Report → Evidence
Runs fully offline on Jetson Orin Nano ($249)

Two HAI-DEF models:
  - CXR Foundation (google/cxr-foundation) — image embeddings + fracture classification
  - MedGemma 1.5 4B (google/medgemma-1.5-4b-it) — clinical report generation
"""

import argparse
import hashlib
import io
import json
import logging
import os
import pickle
import re
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MEDGEMMA_MODEL = os.environ.get("MEDGEMMA_MODEL", "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M")

TRIAGE_THRESHOLDS = {"red": 0.70, "yellow": 0.40}
BODY_REGIONS = ["Hand", "Leg", "Hip", "Shoulder", "Unknown"]

LINEAR_PROBE_PATH = os.environ.get(
    "PROBE_PATH", "results/linear_probe/fracture_probe.joblib"
)
REGION_CLASSIFIER_PATH = os.environ.get(
    "REGION_PATH", "results/linear_probe/region_classifier.joblib"
)
CXR_MODEL_PATH = os.environ.get(
    "CXR_MODEL_PATH", "models/cxr-foundation/elixr-c-v2-pooled"
)

IMAGE_SIZE = 1024  # CXR Foundation input size


# ---------------------------------------------------------------------------
# Prediction Audit Logger (FIX #15)
# ---------------------------------------------------------------------------

class PredictionLogger:
    """JSON-lines audit logger for all predictions.

    Writes one JSON object per line to a rotating log file.  Privacy-safe:
    never logs images or full clinical-context text.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("edgefracture.predictions")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = RotatingFileHandler(
                self.log_dir / "predictions.jsonl",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def log_prediction(
        self,
        result: dict,
        body_region_input: str = "Unknown",
        clinical_context: str = "",
        report_type: str = "",
    ):
        """Record a single prediction to the audit log.

        Args:
            result: The classification result dict from FractureClassifier.classify().
            body_region_input: The body-region value the user originally selected.
            clinical_context: Raw clinical context string (only its *presence*
                is logged, never the content -- for patient privacy).
            report_type: "clinical", "patient", or "" for triage-only calls.
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "probability": result.get("probability"),
            "triage_level": result.get("triage_level"),
            "body_region": result.get("body_region"),
            "body_region_input": body_region_input,
            "region_auto_detected": result.get("region_auto_detected"),
            "model_used": result.get("model_used"),
            "latency_ms": result.get("latency_ms"),
            "confidence": result.get("confidence"),
            "image_warnings": result.get("image_warnings"),
        }
        if report_type:
            entry["report_type"] = report_type
        if clinical_context:
            entry["had_clinical_context"] = True
        self.logger.info(json.dumps(entry))


prediction_logger = PredictionLogger()


# ---------------------------------------------------------------------------
# CXR Foundation + Linear Probe Classifier
# ---------------------------------------------------------------------------

class FractureClassifier:
    """
    Real CXR Foundation inference → linear probe classification.
    Falls back to placeholder mode if models aren't available.
    """

    def __init__(self):
        self.vision_model = None
        self.probe = None
        self.scaler = None
        self.region_classifier = None
        self.model_loaded = False
        self.probe_loaded = False
        self.region_loaded = False
        self._tf = None
        self._try_load()

    # ---- Secure model loading helpers ----

    @staticmethod
    def _verify_hash(model_path: str) -> bool:
        """Verify SHA-256 hash of a model file against its sidecar .sha256 file.

        Returns True if the hash matches or no sidecar exists (first-run
        tolerance). Returns False only when a sidecar exists but the hash
        does not match, indicating the file may have been tampered with.
        """
        hash_path = Path(model_path).with_suffix(
            Path(model_path).suffix + ".sha256"
        )
        if not hash_path.exists():
            print(f"  (no .sha256 sidecar for {model_path} -- skipping verification)")
            return True

        expected = hash_path.read_text().strip()
        actual = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if actual != expected:
            print(f"  SHA-256 MISMATCH for {model_path}!")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
            return False
        return True

    @staticmethod
    def _load_model_file(path: str):
        """Load a model file, preferring .joblib with hash verification,
        falling back to legacy .pkl files for backwards compatibility."""
        p = Path(path)

        # --- Try .joblib (preferred, safer) ---
        if p.suffix == ".joblib" and p.exists():
            if not FractureClassifier._verify_hash(str(p)):
                raise RuntimeError(
                    f"Integrity check failed for {p} -- refusing to load"
                )
            return joblib.load(p), str(p)

        # --- Try .joblib variant if a .pkl path was given ---
        joblib_variant = p.with_suffix(".joblib")
        if joblib_variant.exists():
            if not FractureClassifier._verify_hash(str(joblib_variant)):
                raise RuntimeError(
                    f"Integrity check failed for {joblib_variant} -- refusing to load"
                )
            return joblib.load(joblib_variant), str(joblib_variant)

        # --- Fallback: legacy .pkl (backwards compatibility) ---
        pkl_variant = p.with_suffix(".pkl") if p.suffix != ".pkl" else p
        for fallback in [p, pkl_variant]:
            if fallback.exists():
                print(f"  WARNING: Loading legacy pickle file {fallback}. "
                      f"Re-run training scripts to save as .joblib with hash verification.")
                with open(fallback, "rb") as f:
                    return pickle.load(f), str(fallback)

        raise FileNotFoundError(f"No model file found at {p}, {joblib_variant}, or {pkl_variant}")

    def _try_load(self):
        # --- Load linear probe ---
        self.temperature = None
        if LINEAR_PROBE_PATH:
            try:
                data, actual_path = self._load_model_file(LINEAR_PROBE_PATH)
                if isinstance(data, dict):
                    self.probe = data["model"]
                    self.scaler = data.get("scaler")
                    self.temperature = data.get("temperature")
                else:
                    self.probe = data
                self.probe_loaded = True
                if self.temperature is not None:
                    print(f"Linear probe loaded from {actual_path} "
                          f"(calibrated T={self.temperature:.4f})")
                else:
                    print(f"Linear probe loaded from {actual_path}")
            except FileNotFoundError:
                pass  # Will be reported below
            except Exception as e:
                print(f"Could not load linear probe: {e}")

        # --- Load region classifier ---
        if REGION_CLASSIFIER_PATH:
            try:
                data, actual_path = self._load_model_file(REGION_CLASSIFIER_PATH)
                self.region_classifier = data
                self.region_loaded = True
                print(f"Region classifier loaded from {actual_path}")
                print(f"  Classes: {list(self.region_classifier.classes_)}")
            except FileNotFoundError:
                pass  # Will be reported below
            except Exception as e:
                print(f"Could not load region classifier: {e}")

        # --- Load CXR Foundation vision model ---
        if CXR_MODEL_PATH and os.path.exists(CXR_MODEL_PATH):
            try:
                import tensorflow as tf
                import tensorflow_text  # registers SentencepieceOp needed by CXR Foundation
                self._tf = tf

                # Suppress TF warnings for cleaner output
                tf.get_logger().setLevel("ERROR")
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

                self.vision_model = tf.saved_model.load(CXR_MODEL_PATH)
                self.model_loaded = True
                print(f"CXR Foundation loaded from {CXR_MODEL_PATH}")
            except ImportError:
                print("TensorFlow not available -- CXR Foundation won't load")
            except Exception as e:
                print(f"Could not load CXR Foundation: {e}")

        if not self.model_loaded:
            print("CXR Foundation not loaded -- running in PLACEHOLDER mode")
        if not self.probe_loaded:
            print("Linear probe not loaded -- running in PLACEHOLDER mode")

    # ---- Image preprocessing (matches extraction script) ----

    def _preprocess_image(self, image: Image.Image) -> bytes:
        """Convert PIL image to serialized tf.Example (CXR Foundation input format)."""
        tf = self._tf

        img = image.convert("L")  # Grayscale
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        feature = {
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[png_bytes])
            ),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        return example.SerializeToString()

    def _extract_embedding(self, serialized_example: bytes) -> np.ndarray:
        """Extract raw embedding from CXR Foundation vision encoder."""
        tf = self._tf
        input_tensor = tf.constant([serialized_example])

        try:
            if (hasattr(self.vision_model, "signatures")
                    and "serving_default" in self.vision_model.signatures):
                serve_fn = self.vision_model.signatures["serving_default"]
                input_keys = list(serve_fn.structured_input_signature[1].keys())
                output = serve_fn(**{input_keys[0]: input_tensor})
            else:
                output = self.vision_model(input_tensor)
        except Exception:
            output = self.vision_model(inputs=input_tensor)

        if isinstance(output, dict):
            keys = list(output.keys())
            emb = output[keys[0]].numpy()
            return emb[0].flatten()
        else:
            return output.numpy()[0].flatten()

    # ---- Input validation ----
    # FIX #8: Basic image validation — warn (don't block) if the input
    # doesn't look like an X-ray, to catch accidental uploads of photos, etc.

    @staticmethod
    def _validate_xray_image(image: Image.Image) -> list[str]:
        """Check whether an image looks plausibly like an X-ray.

        Returns a list of warning strings (empty if the image passes all checks).
        This is advisory only — classification still proceeds regardless.
        """
        warnings = []

        # 1. Check image dimensions are reasonable for an X-ray
        #    Typical X-rays are at least a few hundred pixels in each dimension
        #    and not absurdly small (icons) or huge (ultra-high-res photos).
        MIN_DIM = 128
        MAX_DIM = 10000
        w, h = image.size
        if w < MIN_DIM or h < MIN_DIM:
            warnings.append(
                f"Image is very small ({w}x{h}px). X-rays are typically "
                f"at least {MIN_DIM}x{MIN_DIM}px."
            )
        if w > MAX_DIM or h > MAX_DIM:
            warnings.append(
                f"Image is unusually large ({w}x{h}px). This may not be "
                "a standard X-ray image."
            )

        # 2. Check if the image is already grayscale or nearly so.
        #    X-rays are inherently grayscale; a colorful photo is likely not one.
        if image.mode in ("RGB", "RGBA"):
            # Sample pixels to check color saturation
            rgb_image = image.convert("RGB")
            # Downsample for speed
            small = rgb_image.resize((64, 64), Image.NEAREST)
            arr = np.array(small, dtype=np.float32)
            # Compute per-pixel max channel difference as a saturation proxy
            channel_range = arr.max(axis=2) - arr.min(axis=2)  # (64, 64)
            mean_saturation = float(channel_range.mean())
            # Grayscale images have near-zero channel spread; colorful photos
            # typically have mean saturation > 30.
            SATURATION_THRESHOLD = 30.0
            if mean_saturation > SATURATION_THRESHOLD:
                warnings.append(
                    f"Image appears to be a color photograph (mean channel "
                    f"spread={mean_saturation:.1f}). X-rays are typically "
                    "grayscale. Results may be unreliable."
                )

        # 3. Aspect ratio sanity — X-rays are rarely extremely wide or tall
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 5.0:
            warnings.append(
                f"Unusual aspect ratio ({aspect:.1f}:1). This may not be "
                "a standard X-ray image."
            )

        return warnings

    # ---- Classification ----

    def classify(self, image: Image.Image, body_region: str = "Unknown") -> dict:
        start = time.time()

        # Validate that input looks like an X-ray (advisory warnings only)
        image_warnings = self._validate_xray_image(image)

        if self.model_loaded and self.probe_loaded:
            prob, embedding = self._real_classify(image)
            model_used = "CXR Foundation + Linear Probe"

            # Auto-detect body region from the same embedding
            if body_region == "Unknown" and self.region_loaded:
                detected_region, region_conf = self._detect_region(embedding)
                body_region = detected_region
                region_auto = True
            else:
                region_conf = None
                region_auto = (body_region == "Unknown")
        elif self.probe_loaded and not self.model_loaded:
            prob = self._placeholder_score(body_region)
            model_used = "Placeholder (CXR Foundation not loaded)"
            region_conf = None
            region_auto = False
        else:
            prob = self._placeholder_score(body_region)
            model_used = "Placeholder (demo mode)"
            region_conf = None
            region_auto = False

        latency_ms = (time.time() - start) * 1000
        triage_level, triage_color = self._triage(prob)

        result = {
            "probability": round(prob, 3),
            "triage_level": triage_level,
            "triage_color": triage_color,
            "body_region": body_region,
            "confidence": self._confidence_label(prob),
            "model_used": model_used,
            "latency_ms": round(latency_ms, 1),
            "region_auto_detected": region_auto,
        }
        if region_conf is not None:
            result["region_confidence"] = round(region_conf, 3)

        # Attach image validation warnings (FIX #8)
        if image_warnings:
            result["image_warnings"] = image_warnings

        return result

    def _detect_region(self, embedding: np.ndarray) -> tuple[str, float]:
        """Predict body region from CXR Foundation embedding."""
        embedding_2d = embedding.reshape(1, -1)
        region = self.region_classifier.predict(embedding_2d)[0]
        proba = self.region_classifier.predict_proba(embedding_2d)[0]
        conf = float(proba.max())
        # Capitalize to match BODY_REGIONS list
        region = region.capitalize()
        return region, conf

    def _real_classify(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Full pipeline: image → CXR Foundation embedding → linear probe → probability.
        Returns (probability, embedding) so embedding can be reused for region detection."""
        # Step 1: Preprocess
        serialized = self._preprocess_image(image)

        # Step 2: Extract embedding
        embedding = self._extract_embedding(serialized)  # (88064,)

        # Step 3: Linear probe prediction (with temperature calibration)
        embedding_2d = embedding.reshape(1, -1)
        if self.scaler is not None:
            embedding_2d = self.scaler.transform(embedding_2d)
        # Raw logit → temperature-scaled sigmoid for calibrated probabilities.
        # Temperature is loaded from the pkl file (calibrated on held-out set
        # via scipy.optimize.minimize_scalar on log-loss). Falls back to 3.49
        # if not available in the pkl (legacy models).
        logit = self.probe.decision_function(embedding_2d)[0]
        temperature = self.temperature if self.temperature is not None else 3.49
        prob = 1.0 / (1.0 + np.exp(-logit / temperature))

        return float(prob), embedding

    def _placeholder_score(self, body_region: str) -> float:
        demo_scores = {
            "Hand": 0.62, "Leg": 0.78, "Hip": 0.45,
            "Shoulder": 0.33, "Unknown": 0.55,
        }
        return demo_scores.get(body_region, 0.55)

    @staticmethod
    def _triage(prob: float) -> tuple[str, str]:
        if prob >= TRIAGE_THRESHOLDS["red"]:
            return "HIGH SUSPICION", "red"
        elif prob >= TRIAGE_THRESHOLDS["yellow"]:
            return "MODERATE SUSPICION", "yellow"
        else:
            return "LOW SUSPICION", "green"

    @staticmethod
    def _confidence_label(prob: float) -> str:
        if prob > 0.85 or prob < 0.15:
            return "high"
        elif prob > 0.7 or prob < 0.3:
            return "moderate-high"
        else:
            return "moderate"


# ---------------------------------------------------------------------------
# Safety Gate — programmatic safety disclaimers
# ---------------------------------------------------------------------------

_OVERCONFIDENT_PATTERNS = [
    r"\bdiagnos(?:is|ed|e)\b",
    r"\bconfirm(?:s|ed)?\b",
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bclearly shows\b",
    r"\bI can see\b",
    r"\bthe fracture is\b",
    r"\bthis is a\b.*\bfracture\b",
]

SAFETY_DISCLAIMER = (
    "\n\n---\n⚠️ **AI-assisted screening — requires clinician confirmation.** "
    "This is a research prototype, not a diagnostic device. All findings must be "
    "reviewed by a qualified healthcare professional."
)


def safety_gate(text: str) -> str:
    """Scan MedGemma output for overconfident language and append safety disclaimer."""
    # Always append the disclaimer
    text = text.rstrip()

    # Flag if overconfident patterns found
    for pattern in _OVERCONFIDENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(
                pattern,
                lambda m: f"*{m.group(0)}*",  # italicize overconfident terms
                text,
                flags=re.IGNORECASE,
            )

    if not text.endswith(SAFETY_DISCLAIMER.strip()):
        text += SAFETY_DISCLAIMER

    return text


def append_metadata_block(text: str, result: dict) -> str:
    """Programmatically append structured metadata — never rely on MedGemma for this."""
    block = (
        f"\n\n**Screening Metadata**\n"
        f"- CXR Foundation confidence: {round(result['probability'] * 100, 1)}%\n"
        f"- Body region: {result['body_region']}\n"
        f"- Triage level: {result['triage_level']}\n"
        f"- Model: CXR Foundation ELIXR-C v2 → Linear Probe\n"
        f"- Report: MedGemma 1.5 4B (Q4_K_M)\n"
        f"- Inference: {result['latency_ms']}ms (classification)"
    )
    return text + block


# ---------------------------------------------------------------------------
# MedGemma Report Generator
# ---------------------------------------------------------------------------

CLINICAL_PROMPT = """\
You are a musculoskeletal radiology assistant. An AI screening system \
has analyzed a {body_region} X-ray and produced the following results:

- Fracture probability: {score_pct}% ({triage_level})
- Classification: {classification}
- Confidence: {confidence}
{clinical_context_section}
Provide a CLINICAL SUMMARY for the reviewing physician:

1. Describe what this screening result suggests (2-3 sentences), including \
possible fracture types common in this body region.

2. URGENCY LEVEL — classify as one of:
   - URGENT: Likely displaced or unstable fracture — splint and refer immediately
   - MODERATE: Possible non-displaced fracture — further imaging recommended
   - LOW: Low suspicion — clinical correlation advised

3. Recommended next steps (imaging, referral, follow-up).

Be concise and clinical. This is an AI screening result, not a diagnosis."""

PATIENT_PROMPT = """\
You are a friendly medical assistant explaining an X-ray screening result \
to a patient. The AI screening found:

- Body part: {body_region}
- Result: {classification} ({score_pct}% probability)
- Urgency: {triage_level}
{clinical_context_section}
Explain this result in 3-4 sentences using simple, reassuring language. \
Avoid medical jargon. Let them know what happens next and that a doctor \
will review everything. Do not be alarming — be honest but calming."""


# -- Structured JSON prompts ------------------------------------------------

CLINICAL_JSON_PROMPT = """\
You are a musculoskeletal radiology assistant. An AI screening system \
has analyzed a {body_region} X-ray and produced the following results:

- Fracture probability: {score_pct}% ({triage_level})
- Classification: {classification}
- Confidence: {confidence}
{clinical_context_section}
Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) using this exact schema:

{{"primary_finding": "1-2 sentence clinical summary of what the screening suggests",\
 "urgency": "URGENT | MODERATE | LOW",\
 "recommendation": "Recommended next steps (imaging, referral, follow-up)",\
 "differential": "Possible fracture types for this body region",\
 "clinical_note": "Any additional clinical considerations"}}"""

PATIENT_JSON_PROMPT = """\
You are a friendly medical assistant explaining an X-ray screening result \
to a patient. The AI screening found:

- Body part: {body_region}
- Result: {classification} ({score_pct}% probability)
- Urgency: {triage_level}
{clinical_context_section}
Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) using this exact schema:

{{"summary": "2-3 sentence plain-language explanation of the result",\
 "next_steps": "What happens next, in simple terms",\
 "reassurance": "A brief reassuring statement about the process"}}"""

STRUCTURED_JSON_FIELDS_CLINICAL = [
    "primary_finding", "urgency", "recommendation", "differential", "clinical_note",
]
STRUCTURED_JSON_FIELDS_PATIENT = ["summary", "next_steps", "reassurance"]


def _sanitize_clinical_context(text: str) -> str:
    """Sanitize user-provided clinical context before interpolation into LLM prompt.

    FIX #7: Prevents prompt injection by stripping instruction-like patterns,
    truncating to a safe length, and removing prompt-breaking characters.
    """
    # 1. Truncate to 500 characters max
    text = text.strip()[:500]

    # 2. Strip instruction-like patterns that could hijack the LLM prompt
    injection_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?previous",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s*prompt",
        r"override\s+(all\s+)?instructions",
        r"forget\s+(all\s+)?(previous|above|prior)",
        r"act\s+as\s+(a\s+)?",
        r"pretend\s+(you\s+are|to\s+be)",
        r"do\s+not\s+follow",
        r"instead[,:]?\s+(you\s+should|do|say|write)",
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, "[removed]", text, flags=re.IGNORECASE)

    # 3. Remove characters that could break prompt formatting
    #    (keep alphanumeric, basic punctuation, and medical symbols)
    text = re.sub(r"[{}<>|\\`]", "", text)

    return text.strip()


def _build_context_section(clinical_context: str) -> str:
    if clinical_context and clinical_context.strip():
        sanitized = _sanitize_clinical_context(clinical_context)
        if sanitized:
            return f"\nClinical context provided: {sanitized}\n"
    return ""


def _parse_json_response(raw: str) -> dict | None:
    """Try to extract a JSON object from MedGemma's response.

    Handles common LLM quirks: markdown code fences, trailing text, etc.
    Returns the parsed dict or None if parsing fails.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Try the whole string first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find a JSON object within the text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _render_structured_clinical(parsed: dict) -> str:
    """Render parsed clinical JSON into a readable markdown report."""
    lines = []
    if parsed.get("primary_finding"):
        lines.append(f"**Finding:** {parsed['primary_finding']}")
    if parsed.get("urgency"):
        urgency = parsed["urgency"].upper()
        icon = {"URGENT": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(urgency, "⚪")
        lines.append(f"**Urgency:** {icon} {urgency}")
    if parsed.get("differential"):
        lines.append(f"**Differential:** {parsed['differential']}")
    if parsed.get("recommendation"):
        lines.append(f"**Recommendation:** {parsed['recommendation']}")
    if parsed.get("clinical_note"):
        lines.append(f"**Note:** {parsed['clinical_note']}")
    return "\n\n".join(lines)


def _render_structured_patient(parsed: dict) -> str:
    """Render parsed patient JSON into a friendly markdown explanation."""
    lines = []
    if parsed.get("summary"):
        lines.append(parsed["summary"])
    if parsed.get("next_steps"):
        lines.append(f"**What happens next:** {parsed['next_steps']}")
    if parsed.get("reassurance"):
        lines.append(f"*{parsed['reassurance']}*")
    return "\n\n".join(lines)


def _call_medgemma(prompt: str) -> dict:
    """Send a prompt to MedGemma via Ollama and return the raw API response dict.

    Raises requests.ConnectionError, requests.Timeout, or other exceptions
    on failure — callers handle these.
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MEDGEMMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
                "top_p": 0.9,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def generate_report(
    classification_result: dict,
    clinical_context: str = "",
    report_type: str = "clinical",
) -> dict:
    """Generate a structured + narrative report via MedGemma.

    Tries structured JSON output first. On parse failure, falls back to
    the narrative prompt. The returned dict always includes:
      - report_text: markdown narrative (safety-gated + metadata)
      - structured_json: parsed dict or None
      - latency_s: total wall-clock seconds
      - error: error string or None
    """
    prob = classification_result["probability"]
    context_section = _build_context_section(clinical_context)

    fmt_kwargs = dict(
        body_region=classification_result["body_region"],
        score_pct=round(prob * 100, 1),
        triage_level=classification_result["triage_level"],
        classification="Fracture detected" if prob >= 0.5 else "No fracture detected",
        confidence=classification_result["confidence"],
        clinical_context_section=context_section,
    )

    json_template = CLINICAL_JSON_PROMPT if report_type == "clinical" else PATIENT_JSON_PROMPT
    narrative_template = CLINICAL_PROMPT if report_type == "clinical" else PATIENT_PROMPT
    render_fn = _render_structured_clinical if report_type == "clinical" else _render_structured_patient

    start = time.time()
    try:
        # --- Attempt 1: structured JSON ---
        json_prompt = json_template.format(**fmt_kwargs)
        data = _call_medgemma(json_prompt)
        raw_response = data.get("response", "").strip()
        parsed = _parse_json_response(raw_response)

        if parsed is not None:
            # Structured output succeeded — render into narrative
            report_text = render_fn(parsed)
        else:
            # JSON parse failed — fall back to narrative prompt
            narrative_prompt = narrative_template.format(**fmt_kwargs)
            data = _call_medgemma(narrative_prompt)
            report_text = _clean_medgemma_output(data.get("response", "").strip())

        # Apply safety gate and metadata to the narrative
        report_text = safety_gate(report_text)
        report_text = append_metadata_block(report_text, classification_result)

        latency = time.time() - start
        return {
            "report_text": report_text,
            "structured_json": parsed,
            "latency_s": round(latency, 1),
            "error": None,
        }

    except requests.ConnectionError:
        return {
            "report_text": "",
            "structured_json": None,
            "latency_s": 0,
            "error": "Cannot connect to Ollama. Start with: ollama serve",
        }
    except requests.Timeout:
        return {
            "report_text": "",
            "structured_json": None,
            "latency_s": round(time.time() - start, 1),
            "error": "MedGemma timed out (>120s).",
        }
    except Exception as e:
        return {
            "report_text": "",
            "structured_json": None,
            "latency_s": round(time.time() - start, 1),
            "error": f"MedGemma error: {str(e)}",
        }


def _clean_medgemma_output(text: str) -> str:
    """Strip MedGemma artifacts: leaked prompts, critique loops, unused tokens."""
    text = re.sub(r"<unused\d+>", "", text)
    if "You are a musculoskeletal" in text or "You are a friendly" in text:
        for marker in [
            "1.", "CLINICAL SUMMARY", "Clinical Summary",
            "The AI", "Your X-ray", "Based on",
        ]:
            idx = text.find(marker)
            if idx > 0:
                text = text[idx:]
                break
    text = re.sub(r"\*\*Critique:?\*\*.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(
        r"\*\*Self-assessment:?\*\*.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ---------------------------------------------------------------------------
# Evidence / Transparency helpers
# ---------------------------------------------------------------------------

def get_model_transparency_info(result: dict) -> str:
    """Build a transparency explanation of how the AI reached its conclusion."""
    prob = result["probability"]
    region = result["body_region"]

    region_context = {
        "Hand": "metacarpal fractures, phalangeal fractures, boxer's fractures, and scaphoid fractures",
        "Leg": "tibial shaft fractures, fibula fractures, ankle fractures, and stress fractures",
        "Hip": "femoral neck fractures, intertrochanteric fractures, and acetabular fractures",
        "Shoulder": "proximal humerus fractures, clavicle fractures, and scapular fractures",
        "Unknown": "various fracture patterns depending on the anatomical region",
    }

    lines = []
    lines.append("### How This Result Was Produced\n")
    lines.append(
        "**Step 1 — Image Analysis:** Your X-ray was processed by "
        "[CXR Foundation](https://huggingface.co/google/cxr-foundation), "
        "a medical imaging model from Google Health AI. This model was originally "
        "trained on **821,000 chest X-rays** and produces rich visual embeddings "
        "that capture bone and tissue structure.\n"
    )
    lines.append(
        "**Step 2 — Fracture Screening:** The image embeddings (88,064 dimensions) "
        "were passed through a logistic regression classifier trained on the "
        "[FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012) dataset "
        "(4,083 musculoskeletal X-rays across hand, leg, hip, and shoulder). "
        f"The classifier estimated a **{round(prob * 100, 1)}% probability** of fracture.\n"
    )
    lines.append(
        "**Step 3 — Clinical Report:** "
        "[MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it), "
        "a medical language model, interpreted the screening score and generated "
        "a clinical summary. MedGemma **does not see the X-ray image** — it only "
        "receives the numeric score and body region to prevent hallucinated visual findings.\n"
    )
    lines.append("### What This Means\n")

    if prob >= 0.70:
        lines.append(
            f"A score of **{round(prob * 100, 1)}%** indicates high suspicion of fracture. "
            f"Common fractures in the {region.lower()} include {region_context.get(region, '')}. "
            "**This case should be prioritized for radiologist review.**"
        )
    elif prob >= 0.40:
        lines.append(
            f"A score of **{round(prob * 100, 1)}%** falls in the uncertain range. "
            f"The model detected features that partially match fracture patterns in the {region.lower()} "
            f"(common types: {region_context.get(region, '')}). "
            "Additional imaging or clinical correlation is recommended."
        )
    else:
        lines.append(
            f"A score of **{round(prob * 100, 1)}%** suggests low likelihood of fracture. "
            f"However, some fracture types in the {region.lower()} "
            "(e.g., hairline or non-displaced fractures) can be subtle. "
            "Clinical correlation is always advised."
        )

    lines.append("\n### Important Limitations\n")
    lines.append(
        "- CXR Foundation was trained on **chest X-rays only** — its application to "
        "musculoskeletal imaging represents a novel transfer learning approach\n"
        "- This is a **screening tool**, not a diagnostic device\n"
        "- Performance varies by body region and fracture type\n"
        "- Always requires review by a qualified healthcare professional"
    )

    return "\n".join(lines)


def get_performance_context() -> str:
    """Model performance data from actual experiments."""
    lines = []
    lines.append("### Model Performance\n")
    lines.append(
        "CXR Foundation was trained exclusively on chest X-rays, yet demonstrates "
        "strong transfer to musculoskeletal fracture detection — a task it was "
        "**never designed for**.\n"
    )
    lines.append("#### Per-Region AUC (Linear Probe, 5-fold CV)\n")
    lines.append("| Body Region | Images | Fractures | AUC | 95% Bootstrap CI |")
    lines.append("|-------------|--------|-----------|-----|------------------|")
    lines.append("| Hand | 1,510 | 438 | **0.850** | [0.823, 0.876] |")
    lines.append("| Hip | 179 | 10 | **0.864** | [0.706, 0.972] * |")
    lines.append("| Leg | 2,237 | 259 | **0.888** | [0.861, 0.913] |")
    lines.append("| Shoulder | 98 | 10 | **0.848** | [0.667, 0.976] * |")
    lines.append("| **Overall** | **4,024** | **717** | **0.882** | [0.864, 0.899] |")
    lines.append("")
    lines.append(
        "*\\* Hip and Shoulder each have only 10 fracture cases, producing wide "
        "confidence intervals (CI width > 0.15). Their AUC point estimates should "
        "be interpreted with caution — the true AUC could plausibly range from "
        "~0.67 to ~0.98. More labeled data for these regions would substantially "
        "narrow the uncertainty.*"
    )

    lines.append("\n#### Data Efficiency — AUC vs. Training Examples\n")
    lines.append(
        "How many labeled X-rays does it take to make a chest X-ray model "
        "useful for fracture screening?\n"
    )
    lines.append("| Training Examples | AUC | Sensitivity | Specificity |")
    lines.append("|-------------------|-----|-------------|-------------|")
    lines.append("| 10 | 0.556 | 0.004 | 0.999 |")
    lines.append("| 25 | 0.578 | 0.015 | 0.991 |")
    lines.append("| 50 | 0.607 | 0.078 | 0.964 |")
    lines.append("| 100 | 0.683 | 0.191 | 0.947 |")
    lines.append("| 250 | 0.785 | 0.421 | 0.912 |")
    lines.append("| 500 | 0.820 | 0.561 | 0.893 |")
    lines.append("| **4,024 (all)** | **0.882** | **0.692** | **0.917** |")
    lines.append(
        "\n*With just 500 labeled examples, the system crosses the 0.80 AUC threshold "
        "— demonstrating that pre-trained chest X-ray embeddings encode transferable "
        "representations for musculoskeletal fracture detection.*"
    )

    lines.append("\n### Edge Deployment\n")
    lines.append("| Metric | Target | Status |")
    lines.append("|--------|--------|--------|")
    lines.append("| Device | Jetson Orin Nano 8GB | ✓ |")
    lines.append("| Device cost | $249 | ✓ |")
    lines.append("| CXR Foundation latency | < 3s | *pending Jetson benchmarks* |")
    lines.append("| MedGemma report latency | < 90s | *pending Jetson benchmarks* |")
    lines.append("| Total memory (both models) | < 6GB | *pending Jetson benchmarks* |")
    lines.append("| Internet required | None | ✓ Fully offline |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch Triage
# ---------------------------------------------------------------------------

def run_batch_triage(files, body_region):
    """Process multiple X-rays and return a prioritized triage summary table."""
    if not files:
        return "Upload one or more X-ray images to run batch triage."

    rows = []
    for f in files:
        filepath = f.name if hasattr(f, "name") else str(f)
        filename = Path(filepath).name
        try:
            img = Image.open(filepath)
            result = get_classifier().classify(img, body_region)

            # FIX #15: audit-log each batch prediction
            prediction_logger.log_prediction(
                result, body_region_input=body_region,
            )

            color_emoji = {"red": "🔴", "yellow": "🟡", "green": "🟢"}[
                result["triage_color"]
            ]
            rows.append({
                "": color_emoji,
                "File": filename,
                "Fracture %": f"{round(result['probability'] * 100, 1)}%",
                "Triage": result["triage_level"],
                "Region": result["body_region"],
                "Latency": f"{result['latency_ms']}ms",
            })
        except Exception as e:
            rows.append({
                "": "⚠️",
                "File": filename,
                "Fracture %": "Error",
                "Triage": str(e)[:50],
                "Region": body_region,
                "Latency": "-",
            })

    # FIX #1: Sort numerically, not lexicographically. String sort would rank
    # "9.1%" above "78.5%" because "9" > "7" in ASCII — dangerous for triage.
    # Errors (non-numeric values) sort to the end with -1 sentinel.
    rows.sort(
        key=lambda r: float(r["Fracture %"].rstrip("%"))
        if r["Fracture %"] not in ("Error", "-")
        else -1,
        reverse=True,
    )

    if not rows:
        return "No images processed."

    headers = list(rows[0].keys())
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"

    summary_red = sum(1 for r in rows if r[""] == "🔴")
    summary_yellow = sum(1 for r in rows if r[""] == "🟡")
    summary_green = sum(1 for r in rows if r[""] == "🟢")

    md += f"\n**Summary:** {len(rows)} images processed — "
    md += f"🔴 {summary_red} high · 🟡 {summary_yellow} moderate · 🟢 {summary_green} low"

    return md


# ---------------------------------------------------------------------------
# UI State Management
# ---------------------------------------------------------------------------

_classifier = None
_classifier_lock = threading.Lock()


def get_classifier() -> FractureClassifier:
    """Thread-safe lazy singleton for the classifier."""
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:  # double-checked locking
                _classifier = FractureClassifier()
    return _classifier


def check_ollama_status():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.ok:
            models = [m["name"] for m in resp.json().get("models", [])]
            has_medgemma = any("medgemma" in m.lower() for m in models)
            if has_medgemma:
                return "✅ Ollama running · MedGemma available"
            return f"⚠️ Ollama running · MedGemma not found. Models: {', '.join(models) or 'none'}"
        return "⚠️ Ollama error"
    except requests.ConnectionError:
        return "❌ Ollama not running — start with: ollama serve"
    except Exception as e:
        return f"❌ {e}"


def step_triage(image, body_region, clinical_context):
    """Step 1 → Step 2 + Step 3: Run triage and auto-generate both reports."""
    if image is None:
        return (
            gr.update(visible=False), "", "",
            gr.update(visible=False), None,
            "", "", "", "",
        )

    pil_image = (
        Image.fromarray(image) if not isinstance(image, Image.Image) else image
    )
    result = get_classifier().classify(pil_image, body_region)

    # FIX #15: audit-log every triage prediction
    prediction_logger.log_prediction(
        result,
        body_region_input=body_region,
        clinical_context=clinical_context or "",
    )

    state = {
        "result": result,
        "clinical_context": clinical_context or "",
    }

    triage_html = _make_triage_card(result)
    reasoning = _make_reasoning_text(result)

    # Auto-generate both reports
    clinical = generate_report(result, clinical_context or "", report_type="clinical")
    clinical_narrative, clinical_json = _format_report_output(clinical)

    patient = generate_report(result, clinical_context or "", report_type="patient")
    patient_narrative, patient_json = _format_report_output(patient)

    return (
        gr.update(visible=True),
        triage_html,
        reasoning,
        gr.update(visible=True),
        state,
        clinical_narrative,
        clinical_json,
        patient_narrative,
        patient_json,
    )


def _format_report_output(report: dict) -> tuple[str, str]:
    """Format a report dict into (narrative_markdown, raw_json_block).

    Returns a 2-tuple so the UI can render the narrative prominently and
    the raw JSON in a collapsible accordion.
    """
    if report["error"]:
        return f"⚠️ {report['error']}", ""

    narrative = report["report_text"] + (
        f"\n\n---\n*Generated by MedGemma 1.5 4B in {report['latency_s']}s*"
    )

    if report.get("structured_json"):
        raw_json = (
            "```json\n"
            + json.dumps(report["structured_json"], indent=2)
            + "\n```"
        )
    else:
        raw_json = "*Structured output unavailable — narrative fallback used.*"

    return narrative, raw_json


def step_clinical_report(state):
    """Generate clinician-facing report."""
    if not state or "result" not in state:
        return "*Run triage first.*", ""

    report = generate_report(
        state["result"], state.get("clinical_context", ""), report_type="clinical"
    )
    return _format_report_output(report)


def step_patient_report(state):
    """Generate patient-friendly report."""
    if not state or "result" not in state:
        return "*Run triage first.*", ""

    report = generate_report(
        state["result"], state.get("clinical_context", ""), report_type="patient"
    )
    return _format_report_output(report)


def step_show_evidence(state):
    """Load the evidence / transparency panel."""
    if not state or "result" not in state:
        return "*Run triage first.*", "*Run triage first.*"

    transparency = get_model_transparency_info(state["result"])
    performance = get_performance_context()
    return transparency, performance


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _make_triage_card(result: dict) -> str:
    color_map = {
        "red": ("#dc2626", "#fff", "🔴"),
        "yellow": ("#d97706", "#000", "🟡"),
        "green": ("#16a34a", "#fff", "🟢"),
    }
    bg, fg, icon = color_map[result["triage_color"]]
    prob_pct = round(result["probability"] * 100, 1)

    return f"""
    <div style="padding:32px; border-radius:16px; background:{bg}; color:{fg};
                font-family:system-ui; text-align:center; margin:8px 0;">
        <div style="font-size:52px; margin-bottom:8px;">{icon}</div>
        <div style="font-size:24px; font-weight:700; letter-spacing:1px; margin-bottom:4px;">
            {result['triage_level']}
        </div>
        <div style="font-size:48px; font-weight:800; margin:8px 0;">
            {prob_pct}%
        </div>
        <div style="font-size:15px; opacity:0.9;">
            Fracture probability · {result['body_region']} X-ray
        </div>
    </div>
    """


def _make_reasoning_text(result: dict) -> str:
    prob = result["probability"]
    region = result["body_region"]
    model = result["model_used"]
    latency = result["latency_ms"]

    lines = []
    lines.append(
        f"**Classification:** "
        f"{'Fracture detected' if prob >= 0.5 else 'No fracture detected'}"
    )
    lines.append(f"**Confidence:** {result['confidence']}")
    if result.get("region_auto_detected") and result.get("region_confidence"):
        lines.append(
            f"**Body region:** {region} "
            f"*(auto-detected, {round(result['region_confidence'] * 100, 1)}% confidence)*"
        )
    else:
        lines.append(f"**Body region:** {region}")
    lines.append(f"**Model:** {model}")
    lines.append(f"**Latency:** {latency}ms")
    lines.append("")

    if prob >= 0.70:
        lines.append(
            f"⚡ The model detected strong fracture-like patterns in this "
            f"{region.lower()} X-ray. This case should be **prioritized** "
            "for radiologist review."
        )
    elif prob >= 0.40:
        lines.append(
            f"The model detected some features consistent with fracture patterns "
            f"in the {region.lower()}, but with moderate uncertainty. Additional "
            "imaging or clinical correlation is recommended."
        )
    else:
        lines.append(
            f"The model did not detect strong fracture patterns in this "
            f"{region.lower()} X-ray. However, subtle or non-displaced fractures "
            "can be missed. Clinical judgment should always take precedence."
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.disclaimer-box {
    background: #2d1b1b; border: 1px solid #5a2d2d; border-radius: 8px;
    padding: 12px 16px; color: #e8a0a0; font-size: 13px; margin-top: 8px;
}
footer { display: none !important; }
"""


def build_ui():
    with gr.Blocks(
        title="EdgeFracture — Fracture Triage",
    ) as app:

        triage_state = gr.State(value=None)

        # ── Header ──
        gr.Markdown(
            """
            # 🦴 EdgeFracture
            **Portable Musculoskeletal Fracture Triage on Edge Hardware**
            *CXR Foundation + MedGemma · Fully offline on Jetson Orin Nano ($249)*
            """
        )

        with gr.Tabs():
            # ==============================================================
            # TAB 1: Single Image Triage (Stepped)
            # ==============================================================
            with gr.Tab("🔍 Triage", id="triage"):

                # ── STEP 1: Upload & Context ──
                gr.Markdown("#### Step 1 — Upload X-ray & Provide Context")
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="X-ray Image", type="pil", height=350,
                        )
                    with gr.Column(scale=1):
                        body_region = gr.Dropdown(
                            choices=BODY_REGIONS, value="Unknown",
                            label="Body Region",
                            info="Leave as 'Unknown' to auto-detect, or select manually to override",
                        )
                        clinical_context = gr.Textbox(
                            label="Clinical Context (optional)",
                            placeholder=(
                                "e.g., 'Patient fell on outstretched hand, "
                                "point tenderness over anatomical snuffbox, 45yo female'"
                            ),
                            lines=3,
                        )
                        triage_btn = gr.Button(
                            "🔍 Analyze X-ray", variant="primary", size="lg",
                        )

                # ── STEP 2: Triage Result ──
                step2_container = gr.Column(visible=False)
                with step2_container:
                    gr.Markdown("---")
                    gr.Markdown("#### Step 2 — Triage Result")
                    with gr.Row():
                        with gr.Column(scale=1):
                            triage_card = gr.HTML()
                        with gr.Column(scale=1):
                            reasoning_text = gr.Markdown()

                # ── STEP 3: Clinical Reports ──
                step3_container = gr.Column(visible=False)
                with step3_container:
                    gr.Markdown("---")
                    gr.Markdown("#### Step 3 — AI Clinical Report")
                    gr.Markdown(
                        "*MedGemma automatically generates reports for clinicians "
                        "and patients after triage. Clinical context (if provided "
                        "above) is included in the prompt.*"
                    )

                    with gr.Tabs():
                        with gr.Tab("🩺 Clinician View"):
                            clinical_report_md = gr.Markdown(
                                "*Report will appear after triage analysis.*"
                            )
                            with gr.Accordion(
                                "📦 Structured JSON (machine-readable)", open=False,
                            ):
                                clinical_json_md = gr.Markdown("")
                        with gr.Tab("💬 Patient View"):
                            patient_report_md = gr.Markdown(
                                "*Report will appear after triage analysis.*"
                            )
                            with gr.Accordion(
                                "📦 Structured JSON (machine-readable)", open=False,
                            ):
                                patient_json_md = gr.Markdown("")

                # ── STEP 4: Evidence & Transparency ──
                gr.Markdown("---")
                with gr.Accordion(
                    "📊 Evidence & Transparency — How It Works", open=False,
                ):
                    evidence_btn = gr.Button(
                        "Load Evidence Panel", variant="secondary", size="sm",
                    )
                    with gr.Tabs():
                        with gr.Tab("🔬 This Result"):
                            evidence_this_md = gr.Markdown(
                                "*Click 'Load Evidence Panel' to see how "
                                "this result was produced.*"
                            )
                        with gr.Tab("📈 Model Performance"):
                            evidence_perf_md = gr.Markdown(
                                "*Click 'Load Evidence Panel' to see "
                                "model benchmarks and data efficiency results.*"
                            )

                # ── Disclaimer ──
                gr.HTML("""
                <div class="disclaimer-box">
                    <strong>⚠️ Research Prototype — Not for Clinical Use</strong><br>
                    EdgeFracture is an AI-assisted screening tool for research and
                    demonstration only. Not FDA-cleared. Not a diagnostic device.
                    All outputs require review by a qualified healthcare professional.
                    AI screening supplements but never replaces clinical judgment.
                </div>
                """)

            # ==============================================================
            # TAB 2: Batch Triage
            # ==============================================================
            with gr.Tab("📋 Batch Triage", id="batch"):
                gr.Markdown(
                    """
                    #### Batch Triage Queue
                    Upload multiple X-ray images to get a prioritized triage
                    summary. Cases are **sorted by risk score** (highest first)
                    — simulating a real ED workflow where the most urgent cases
                    surface to the top.
                    """
                )
                with gr.Row():
                    batch_files = gr.File(
                        label="Upload X-ray Images",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    batch_region = gr.Dropdown(
                        choices=BODY_REGIONS, value="Unknown",
                        label="Body Region (applied to all)",
                    )
                batch_btn = gr.Button(
                    "🔍 Run Batch Triage", variant="primary", size="lg",
                )
                batch_results = gr.Markdown(
                    "*Upload images and click Run Batch Triage.*"
                )

            # ==============================================================
            # TAB 3: About / System
            # ==============================================================
            with gr.Tab("ℹ️ About", id="about"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """
                            ### EdgeFracture

                            **A portable, fully offline fracture screening station**
                            that repurposes a chest X-ray foundation model to triage
                            musculoskeletal fractures across body regions — running on
                            a $249 Jetson Orin Nano.

                            ---

                            **The Problem:** Missed fractures are the #1 diagnostic
                            error in emergency departments (3–10% miss rate).
                            4.3 billion people live in regions with fewer than
                            1 radiologist per 100,000 population.

                            **Our Approach:** Google's
                            [CXR Foundation](https://huggingface.co/google/cxr-foundation),
                            trained only on chest X-rays, transfers to musculoskeletal
                            fracture detection across 4 body regions — with as few as
                            500 labeled examples reaching 0.82 AUC.
                            [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it)
                            provides clinical reasoning and patient communication.

                            **Key Result:** Overall AUC **0.882** across hand, leg,
                            hip, and shoulder — from a model that has never seen a
                            musculoskeletal X-ray during training.

                            ---

                            **HAI-DEF Models Used:**
                            - **CXR Foundation** — Image embeddings & fracture classification
                            - **MedGemma 1.5 4B** — Clinical report generation

                            **Dataset:**
                            [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012)
                            — 4,083 MSK X-rays · CC BY 4.0

                            **Hardware:** NVIDIA Jetson Orin Nano 8GB · $249 · Fully offline

                            **Competition:**
                            [MedGemma Impact Challenge 2026](https://www.kaggle.com/competitions/medgemma-impact-challenge)
                            """
                        )
                    with gr.Column():
                        gr.Markdown("### System Status")
                        status_text = gr.Textbox(
                            value=check_ollama_status(),
                            label="Ollama / MedGemma",
                            interactive=False,
                        )
                        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

                        gr.Markdown(
                            f"""
                            ### Configuration
                            - **Ollama URL:** `{OLLAMA_BASE_URL}`
                            - **MedGemma:** `{MEDGEMMA_MODEL}`
                            - **CXR Foundation:** `{'Loaded ✓' if get_classifier().model_loaded else 'Not loaded — placeholder mode'}`
                            - **Linear probe:** `{'Loaded ✓ (0.882 AUC)' if get_classifier().probe_loaded else 'Not loaded — placeholder mode'}`
                            - **Region classifier:** `{'Loaded ✓ — auto-detects body region' if get_classifier().region_loaded else 'Not loaded — manual selection'}`
                            """
                        )

        # ── Event wiring ──

        triage_btn.click(
            fn=step_triage,
            inputs=[image_input, body_region, clinical_context],
            outputs=[
                step2_container, triage_card, reasoning_text,
                step3_container, triage_state,
                clinical_report_md, clinical_json_md,
                patient_report_md, patient_json_md,
            ],
        )

        evidence_btn.click(
            fn=step_show_evidence,
            inputs=[triage_state],
            outputs=[evidence_this_md, evidence_perf_md],
        )

        batch_btn.click(
            fn=run_batch_triage,
            inputs=[batch_files, batch_region],
            outputs=[batch_results],
        )

        refresh_btn.click(
            fn=check_ollama_status,
            inputs=[],
            outputs=[status_text],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFracture Triage App")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  EdgeFracture — Portable Fracture Triage")
    print("=" * 60)
    print(f"  Ollama:       {OLLAMA_BASE_URL}")
    print(f"  MedGemma:     {MEDGEMMA_MODEL}")
    print(f"  CXR Model:    {CXR_MODEL_PATH or 'Not configured'}")
    print(f"  Linear Probe: {LINEAR_PROBE_PATH or 'Not configured'}")
    print(f"  Region Model: {REGION_CLASSIFIER_PATH or 'Not configured'}")
    print(f"  CXR loaded:   {'✓' if get_classifier().model_loaded else '✗ (placeholder mode)'}")
    print(f"  Probe loaded: {'✓' if get_classifier().probe_loaded else '✗ (placeholder mode)'}")
    print(f"  Region loaded:{'✓ (auto-detect)' if get_classifier().region_loaded else '✗ (manual select)'}")
    print("=" * 60)

    app = build_ui()
    app.launch(
        server_name=args.host, server_port=args.port, share=args.share,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )