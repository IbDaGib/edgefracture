"""CXR Foundation + linear probe classifier implementation."""

import hashlib
import io
import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

try:
    from .config import (
        CXR_MODEL_PATH,
        IMAGE_SIZE,
        LINEAR_PROBE_PATH,
        TRIAGE_THRESHOLDS,
    )
except ImportError:
    from config import CXR_MODEL_PATH, IMAGE_SIZE, LINEAR_PROBE_PATH, TRIAGE_THRESHOLDS


class FractureClassifier:
    """
    Real CXR Foundation inference -> linear probe classification.
    Falls back to placeholder mode if models aren't available.
    """

    def __init__(self):
        self.vision_model = None
        self.probe = None
        self.scaler = None
        self.model_loaded = False
        self.probe_loaded = False
        self._tf = None
        self._try_load()

    @staticmethod
    def _verify_hash(model_path: str) -> bool:
        """Verify SHA-256 hash of a model file against its sidecar .sha256 file."""
        hash_path = Path(model_path).with_suffix(Path(model_path).suffix + ".sha256")
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
        """Load model file preferring .joblib with hash verification."""
        p = Path(path)

        if p.suffix == ".joblib" and p.exists():
            if not FractureClassifier._verify_hash(str(p)):
                raise RuntimeError(f"Integrity check failed for {p} -- refusing to load")
            return joblib.load(p), str(p)

        joblib_variant = p.with_suffix(".joblib")
        if joblib_variant.exists():
            if not FractureClassifier._verify_hash(str(joblib_variant)):
                raise RuntimeError(
                    f"Integrity check failed for {joblib_variant} -- refusing to load"
                )
            return joblib.load(joblib_variant), str(joblib_variant)

        pkl_variant = p.with_suffix(".pkl") if p.suffix != ".pkl" else p
        for fallback in [p, pkl_variant]:
            if fallback.exists():
                print(
                    f"  WARNING: Loading legacy pickle file {fallback}. "
                    f"Re-run training scripts to save as .joblib with hash verification."
                )
                with open(fallback, "rb") as f:
                    return pickle.load(f), str(fallback)

        raise FileNotFoundError(
            f"No model file found at {p}, {joblib_variant}, or {pkl_variant}"
        )

    def _try_load(self):
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
                    print(
                        f"Linear probe loaded from {actual_path} "
                        f"(calibrated T={self.temperature:.4f})"
                    )
                else:
                    print(f"Linear probe loaded from {actual_path}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Could not load linear probe: {e}")

        if CXR_MODEL_PATH and os.path.exists(CXR_MODEL_PATH):
            try:
                import tensorflow as tf
                import tensorflow_text

                self._tf = tf
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

    def _preprocess_image(self, image: Image.Image) -> bytes:
        """Convert PIL image to serialized tf.Example."""
        tf = self._tf

        img = image.convert("L")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        feature = {
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[png_bytes])
            ),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def _extract_embedding(self, serialized_example: bytes) -> np.ndarray:
        """Extract raw embedding from CXR Foundation vision encoder."""
        tf = self._tf
        input_tensor = tf.constant([serialized_example])

        try:
            if (
                hasattr(self.vision_model, "signatures")
                and "serving_default" in self.vision_model.signatures
            ):
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
        return output.numpy()[0].flatten()

    @staticmethod
    def _validate_xray_image(image: Image.Image) -> list[str]:
        """Check whether an image looks plausibly like an X-ray."""
        warnings = []

        min_dim = 128
        max_dim = 10000
        w, h = image.size
        if w < min_dim or h < min_dim:
            warnings.append(
                f"Image is very small ({w}x{h}px). X-rays are typically "
                f"at least {min_dim}x{min_dim}px."
            )
        if w > max_dim or h > max_dim:
            warnings.append(
                f"Image is unusually large ({w}x{h}px). This may not be "
                "a standard X-ray image."
            )

        if image.mode in ("RGB", "RGBA"):
            rgb_image = image.convert("RGB")
            small = rgb_image.resize((64, 64), Image.NEAREST)
            arr = np.array(small, dtype=np.float32)
            channel_range = arr.max(axis=2) - arr.min(axis=2)
            mean_saturation = float(channel_range.mean())
            saturation_threshold = 30.0
            if mean_saturation > saturation_threshold:
                warnings.append(
                    f"Image appears to be a color photograph (mean channel "
                    f"spread={mean_saturation:.1f}). X-rays are typically "
                    "grayscale. Results may be unreliable."
                )

        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 5.0:
            warnings.append(
                f"Unusual aspect ratio ({aspect:.1f}:1). This may not be "
                "a standard X-ray image."
            )

        return warnings

    def classify(self, image: Image.Image, body_region: str = "Unknown") -> dict:
        start = time.time()
        image_warnings = self._validate_xray_image(image)

        region_auto = False
        if self.model_loaded and self.probe_loaded:
            prob, _embedding = self._real_classify(image)
            model_used = "CXR Foundation + Linear Probe"
        elif self.probe_loaded and not self.model_loaded:
            prob = self._placeholder_score(body_region)
            model_used = "Placeholder (CXR Foundation not loaded)"
        else:
            prob = self._placeholder_score(body_region)
            model_used = "Placeholder (demo mode)"

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
        if image_warnings:
            result["image_warnings"] = image_warnings
        return result

    def _real_classify(self, image: Image.Image) -> tuple[float, np.ndarray]:
        serialized = self._preprocess_image(image)
        embedding = self._extract_embedding(serialized)

        embedding_2d = embedding.reshape(1, -1)
        if self.scaler is not None:
            embedding_2d = self.scaler.transform(embedding_2d)

        logit = self.probe.decision_function(embedding_2d)[0]
        temperature = self.temperature if self.temperature is not None else 3.49
        prob = 1.0 / (1.0 + np.exp(-logit / temperature))
        return float(prob), embedding

    def _placeholder_score(self, body_region: str) -> float:
        demo_scores = {
            "Hand": 0.62,
            "Leg": 0.78,
            "Hip": 0.45,
            "Shoulder": 0.33,
            "Unknown": 0.55,
        }
        return demo_scores.get(body_region, 0.55)

    @staticmethod
    def _triage(prob: float) -> tuple[str, str]:
        if prob >= TRIAGE_THRESHOLDS["red"]:
            return "HIGH SUSPICION", "red"
        if prob >= TRIAGE_THRESHOLDS["yellow"]:
            return "MODERATE SUSPICION", "yellow"
        return "LOW SUSPICION", "green"

    @staticmethod
    def _confidence_label(prob: float) -> str:
        if prob > 0.85 or prob < 0.15:
            return "high"
        if prob > 0.7 or prob < 0.3:
            return "moderate-high"
        return "moderate"
