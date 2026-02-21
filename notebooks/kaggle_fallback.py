#!/usr/bin/env python3
"""
EdgeFracture — Kaggle T4 Fallback Notebook

Use this if CXR Foundation doesn't load on Jetson ARM64.
Run on Kaggle with T4 GPU to extract embeddings, then copy
the .npy files to the Jetson for the rest of the pipeline.

Steps:
  1. Upload this to Kaggle as a notebook
  2. Enable GPU (T4)
  3. Add FracAtlas dataset
  4. Run all cells
  5. Download embeddings.npy + metadata.csv
  6. Copy to Jetson: results/embeddings/

This is a CLEAN NOTEBOOK ready for Kaggle submission — it also
serves as your reproducible code artifact for the competition.
"""

# ============================================================
# Cell 1: Setup
# ============================================================

# !pip install -q transformers huggingface_hub safetensors pillow tqdm scikit-learn matplotlib

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Cell 2: Load CXR Foundation
# ============================================================

# Login to HuggingFace (gated model)
from huggingface_hub import login
# login(token="YOUR_HF_TOKEN")  # Uncomment and add your token

MODEL_ID = "google/cxr-foundation"

print(f"Loading {MODEL_ID}...")
from transformers import AutoModel, AutoProcessor

# NOTE: Adapt this based on the actual model packaging.
# CXR Foundation may use trust_remote_code=True
try:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(DEVICE).eval()
    print(f"Model loaded: {type(model)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
except Exception as e:
    print(f"AutoModel failed: {e}")
    print("Trying alternative loading methods...")
    # Add alternative loading here based on model card instructions

# ============================================================
# Cell 3: Load FracAtlas Dataset
# ============================================================

# On Kaggle, the dataset path depends on how you added it
# Common paths: /kaggle/input/fracatlas/ or similar
FRACATLAS_DIR = Path("/kaggle/input/fracatlas")  # Adjust this

if not FRACATLAS_DIR.exists():
    # Try to find it
    for p in Path("/kaggle/input").iterdir():
        print(f"  Found: {p}")
    print("Update FRACATLAS_DIR above to match your dataset path")

# Find all images
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
all_images = sorted([
    f for f in FRACATLAS_DIR.rglob("*") 
    if f.suffix.lower() in IMAGE_EXTS
])
print(f"Found {len(all_images)} images")

# ============================================================
# Cell 4: Extract Embeddings
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((1280, 1280)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

embeddings = []
metadata = []
timings = []
errors = []

for img_path in tqdm(all_images, desc="Extracting embeddings"):
    try:
        img = Image.open(img_path).convert("L")
        img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(img_tensor)
            # Adapt based on actual output format:
            if hasattr(outputs, "image_embeds"):
                emb = outputs.image_embeds
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1)
            elif isinstance(outputs, dict):
                emb = list(outputs.values())[0]
            else:
                emb = outputs
        t1 = time.perf_counter()
        
        emb_np = emb.cpu().numpy().flatten()
        embeddings.append(emb_np)
        timings.append(t1 - t0)
        
        # Infer body region and fracture status from path
        parts = [p.lower() for p in img_path.parts]
        has_fracture = 1 if any("fracture" in p for p in parts) else 0
        
        if any("hand" in p or "finger" in p or "wrist" in p for p in parts):
            region = "hand"
        elif any("leg" in p or "ankle" in p or "knee" in p or "tibia" in p for p in parts):
            region = "leg"
        elif any("hip" in p or "pelvis" in p for p in parts):
            region = "hip"
        elif any("shoulder" in p or "humerus" in p for p in parts):
            region = "shoulder"
        else:
            region = "unknown"
        
        metadata.append({
            "file_name": img_path.name,
            "file_path": str(img_path.relative_to(FRACATLAS_DIR)),
            "body_region": region,
            "has_fracture": has_fracture,
        })
        
    except Exception as e:
        errors.append({"file": str(img_path), "error": str(e)})

print(f"\nExtracted: {len(embeddings)} embeddings")
print(f"Errors: {len(errors)}")
print(f"Mean latency: {np.mean(timings)*1000:.1f} ms")

# ============================================================
# Cell 5: Save Embeddings
# ============================================================

embeddings_array = np.array(embeddings)
metadata_df = pd.DataFrame(metadata)

# Save to /kaggle/working for download
output_dir = Path("/kaggle/working")
np.save(output_dir / "embeddings.npy", embeddings_array)
metadata_df.to_csv(output_dir / "metadata.csv", index=False)

print(f"Embeddings shape: {embeddings_array.shape}")
print(f"Saved to {output_dir}")
print(f"\nFracture distribution:")
print(metadata_df["has_fracture"].value_counts())
print(f"\nBody region distribution:")
print(metadata_df["body_region"].value_counts())

# ============================================================
# Cell 6: Quick Validation — Linear Probe
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

mask = metadata_df["has_fracture"].isin([0, 1])
X = embeddings_array[mask.values]
y = metadata_df.loc[mask, "has_fracture"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

print(f"\n{'='*50}")
print(f"LINEAR PROBE VALIDATION")
print(f"{'='*50}")
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"Per-fold: {[f'{s:.4f}' for s in scores]}")

# ============================================================
# Cell 7: Data Efficiency Curve
# ============================================================

import matplotlib.pyplot as plt

subset_sizes = [10, 25, 50, 100, 250, 500, len(X)]
results = []

for n in subset_sizes:
    fold_aucs = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]
        
        if n < len(X_train):
            rng = np.random.RandomState(42 + fold)
            idx = rng.choice(len(X_train), min(n, len(X_train)), replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]
        
        try:
            fold_aucs.append(roc_auc_score(y_test, y_score))
        except:
            pass
    
    results.append({
        "n": n,
        "mean_auc": np.mean(fold_aucs),
        "std_auc": np.std(fold_aucs),
    })
    print(f"n={n:>5}: AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sizes = [r["n"] for r in results]
means = [r["mean_auc"] for r in results]
stds = [r["std_auc"] for r in results]

ax.plot(sizes, means, "o-", color="#2c3e50", linewidth=2.5, markersize=8)
ax.fill_between(sizes, [m-s for m,s in zip(means,stds)],
                [m+s for m,s in zip(means,stds)], alpha=0.2, color="#3498db")
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Clinical target")
ax.set_xscale("log")
ax.set_xlabel("Labeled Training Examples")
ax.set_ylabel("AUC (5-Fold CV)")
ax.set_title("CXR Foundation → MSK Fracture Detection: Data Efficiency")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(output_dir / "data_efficiency_curve.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {output_dir / 'data_efficiency_curve.png'}")

# ============================================================
# Cell 8: Save Final Model for Jetson
# ============================================================

import pickle

clf_final = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
clf_final.fit(X_scaled, y)

with open(output_dir / "fracture_probe.pkl", "wb") as f:
    pickle.dump({"model": clf_final, "scaler": scaler}, f)

print(f"Deployment model saved: {output_dir / 'fracture_probe.pkl'}")
print(f"\nCopy these files to your Jetson:")
print(f"  embeddings.npy → results/embeddings/")
print(f"  metadata.csv → results/embeddings/")
print(f"  fracture_probe.pkl → results/linear_probe/")
print(f"  data_efficiency_curve.png → results/linear_probe/")
