"""
EdgeFracture — Real Classifier Integration

Drop-in replacement for the placeholder classifier in app.py.
Use this once you have:
  1. CXR Foundation model (TF SavedModel or converted weights)
  2. Trained linear probe (fracture_probe.pkl from experiment 3)

Usage:
    # In app.py, replace:
    #   classifier = FractureClassifier()
    # with:
    #   from classifier_real import RealFractureClassifier
    #   classifier = RealFractureClassifier("path/to/cxr-foundation", "path/to/fracture_probe.pkl")
"""

import pickle
import time

import numpy as np
from PIL import Image


class RealFractureClassifier:
    """
    CXR Foundation embeddings → linear probe → fracture probability.
    
    This handles TWO embedding formats since CXR Foundation can return either:
      - ELIXR-contrastive: shape (32, 128) → 4,096-dim (from the contrastive head)
      - Raw feature maps: shape (8, 8, 1376) → 88,064-dim (from the backbone)
    
    Your linear probe was trained on one of these — make sure they match.
    """

    def __init__(self, cxr_model_path: str, probe_path: str, device: str = "cpu"):
        import torch

        self.device = device

        # Load linear probe
        with open(probe_path, "rb") as f:
            self.probe = pickle.load(f)
        print(f"✓ Linear probe loaded ({type(self.probe).__name__})")

        # --- OPTION A: TensorFlow SavedModel (if running TF) ---
        # Uncomment this block if your CXR Foundation is TF SavedModel format
        #
        # import tensorflow as tf
        # self.cxr_model = tf.saved_model.load(cxr_model_path)
        # self.use_tf = True
        # print(f"✓ CXR Foundation loaded (TensorFlow)")

        # --- OPTION B: Pre-extracted embeddings only (no live model) ---
        # If CXR Foundation won't load on Jetson ARM64, load pre-computed embeddings
        # and match by filename
        #
        # self.embeddings = np.load(os.path.join(cxr_model_path, "embeddings.npy"))
        # self.metadata = pd.read_csv(os.path.join(cxr_model_path, "metadata.csv"))
        # self.use_precomputed = True

        self.model_loaded = True
        self.probe_loaded = True

    def classify(self, image: Image.Image, body_region: str = "Unknown") -> dict:
        start = time.time()

        # Extract embedding
        embedding = self._extract_embedding(image)

        # Run linear probe
        prob = float(self.probe.predict_proba(embedding.reshape(1, -1))[0, 1])

        latency_ms = (time.time() - start) * 1000

        triage_level, triage_color = self._triage(prob)

        return {
            "probability": round(prob, 3),
            "triage_level": triage_level,
            "triage_color": triage_color,
            "body_region": body_region,
            "confidence": "moderate" if 0.3 < prob < 0.7 else "high",
            "model_used": "CXR Foundation + Linear Probe",
            "latency_ms": round(latency_ms, 1),
        }

    def _extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image and extract CXR Foundation embeddings.
        
        CXR Foundation expects:
          - Input: 1280×1280 grayscale image
          - Pixel values: normalized (check the model card for exact normalization)
          - Output: depends on which head you use
        
        Your current extraction gives (8, 8, 1376) = 88,064-dim.
        Flatten that to match what the linear probe expects.
        """
        # Resize to CXR Foundation input size
        img = image.convert("L").resize((1280, 1280), Image.BILINEAR)

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # --- TensorFlow inference ---
        # import tensorflow as tf
        # input_tensor = tf.constant(img_array[np.newaxis, :, :, np.newaxis])
        # outputs = self.cxr_model(input_tensor)
        # 
        # # The output dict has multiple heads — use whichever your probe was trained on
        # # Common keys: 'feature_maps_0', 'feature_maps_1', 'contrastive', etc.
        # embedding = outputs['feature_maps_0'].numpy()  # shape (1, 8, 8, 1376)
        # embedding = embedding.flatten()  # → (88064,)
        # return embedding

        raise NotImplementedError(
            "Uncomment the TF or PyTorch extraction code above, "
            "matching whichever format your linear probe was trained on."
        )

    @staticmethod
    def _triage(prob: float) -> tuple[str, str]:
        if prob >= 0.70:
            return "HIGH SUSPICION", "red"
        elif prob >= 0.40:
            return "MODERATE SUSPICION", "yellow"
        else:
            return "LOW SUSPICION", "green"


class PrecomputedClassifier:
    """
    Fallback for Jetson if CXR Foundation won't load on ARM64.
    
    Uses pre-extracted embeddings from Mac → matches uploaded images
    against the FracAtlas test set by filename, then runs the linear probe.
    
    For the DEMO: load a few known FracAtlas images and show real scores.
    For NEW images: falls back to placeholder (can't extract embeddings without model).
    """

    def __init__(self, embeddings_path: str, metadata_path: str, probe_path: str):
        import pandas as pd

        self.embeddings = np.load(embeddings_path)  # shape (N, D)
        self.metadata = pd.read_csv(metadata_path)
        with open(probe_path, "rb") as f:
            self.probe = pickle.load(f)

        # Build filename → index lookup
        self.filename_to_idx = {
            row["filename"]: i for i, row in self.metadata.iterrows()
        }
        print(f"✓ Precomputed classifier: {len(self.filename_to_idx)} images indexed")

        self.model_loaded = True
        self.probe_loaded = True

    def classify(self, image: Image.Image, body_region: str = "Unknown",
                 filename: str = "") -> dict:
        start = time.time()

        if filename in self.filename_to_idx:
            idx = self.filename_to_idx[filename]
            embedding = self.embeddings[idx]
            prob = float(self.probe.predict_proba(embedding.reshape(1, -1))[0, 1])
            model_used = "CXR Foundation (pre-computed) + Linear Probe"
        else:
            # Can't classify unknown images without live model
            prob = 0.5
            model_used = "Placeholder (image not in pre-computed set)"

        latency_ms = (time.time() - start) * 1000
        triage_level, triage_color = self._triage(prob)

        return {
            "probability": round(prob, 3),
            "triage_level": triage_level,
            "triage_color": triage_color,
            "body_region": body_region,
            "confidence": "moderate" if 0.3 < prob < 0.7 else "high",
            "model_used": model_used,
            "latency_ms": round(latency_ms, 1),
        }

    @staticmethod
    def _triage(prob: float) -> tuple[str, str]:
        if prob >= 0.70:
            return "HIGH SUSPICION", "red"
        elif prob >= 0.40:
            return "MODERATE SUSPICION", "yellow"
        else:
            return "LOW SUSPICION", "green"