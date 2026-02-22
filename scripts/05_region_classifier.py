#!/usr/bin/env python3
"""
Train a body region classifier on CXR Foundation embeddings.

Uses the same embeddings extracted in 01_extract_embeddings.py to predict
which body region (hand, leg, hip, shoulder) an X-ray belongs to.

This runs instantly alongside the fracture probe — zero extra latency.

Usage:
    python scripts/05_region_classifier.py
"""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_PATH = Path("results/embeddings/embeddings.npy")
METADATA_PATH = Path("results/embeddings/metadata.csv")
OUTPUT_DIR = Path("results/linear_probe")


def main():
    # ---- Load data ----
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
    print(f"Regions: {dict(metadata['body_region'].value_counts())}")

    X = embeddings
    y = metadata["body_region"].values

    # ---- Train with cross-validation ----
    print("\n" + "=" * 60)
    print("BODY REGION CLASSIFIER (5-fold CV)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # StandardScaler + LogisticRegression pipeline — matches the approach
    # used in the fracture probe for consistent preprocessing of embeddings.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\nAccuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Per-fold: {[f'{s:.4f}' for s in scores]}")

    # ---- Per-class report (cross-validated, out-of-fold predictions) ----
    y_pred = cross_val_predict(model, X, y, cv=cv)

    print(f"\nClassification Report (cross-validated):")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    labels = sorted(metadata["body_region"].unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    print(f"{'':>12s}", "  ".join(f"{l:>8s}" for l in labels))
    for i, label in enumerate(labels):
        print(f"{label:>12s}", "  ".join(f"{cm[i,j]:>8d}" for j in range(len(labels))))

    # ---- Train final deployment model on all data and save (joblib + SHA-256) ----
    import joblib

    model.fit(X, y)
    output_path = OUTPUT_DIR / "region_classifier.joblib"
    joblib.dump(model, output_path)

    # Write SHA-256 sidecar for integrity verification on load
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    hash_path = output_path.with_suffix(output_path.suffix + ".sha256")
    hash_path.write_text(sha256 + "\n")

    size_kb = output_path.stat().st_size / 1024
    print(f"\nSaved: {output_path} ({size_kb:.1f} KB)")
    print(f"  Hash: {hash_path} ({sha256[:16]}...)")
    print(f"Classes: {list(model.classes_)}")

    # ---- Save summary ----
    summary = {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "per_fold": [float(s) for s in scores],
        "classes": list(model.classes_),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "region_counts": {k: int(v) for k, v in metadata["body_region"].value_counts().items()},
    }
    with open(OUTPUT_DIR / "region_classifier_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Done — use region_classifier.joblib alongside fracture_probe.joblib")

if __name__ == "__main__":
    main()
