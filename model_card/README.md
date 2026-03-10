---
license: apache-2.0
base_model: google/cxr-foundation
tags:
  - medical-imaging
  - radiology
  - fracture-detection
  - linear-probe
  - cxr-foundation
  - musculoskeletal
  - x-ray
  - edge-ai
  - jetson
pipeline_tag: image-classification
datasets:
  - FracAtlas
metrics:
  - roc_auc
library_name: sklearn
---

# EdgeFracture CXR Fracture Probe

A lightweight linear probe that classifies fractures from [CXR Foundation](https://developers.google.com/health/imaging/cxr-foundation) embeddings. Designed for edge deployment on NVIDIA Jetson Orin Nano (8 GB).

## Model Description

This is a logistic regression classifier trained on 88,064-dimensional embeddings extracted by Google's CXR Foundation model. It converts CXR Foundation — a chest X-ray foundation model — into a musculoskeletal fracture detector, demonstrating that CXR Foundation's learned representations transfer beyond its original thoracic training domain.

- **Architecture:** `sklearn.linear_model.LogisticRegression` with `StandardScaler` preprocessing and temperature calibration (T=3.49)
- **Input:** 88,064-dim CXR Foundation embedding vector
- **Output:** Fracture probability (0–1)
- **File format:** joblib dict with keys `model`, `scaler`, `temperature`
- **Size:** ~700 KB

## Training Data

Trained on [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012) — 4,083 musculoskeletal X-rays (717 fractures, 3,366 normal) across four body regions: hand, leg, hip, and shoulder.

## Performance

5-fold stratified cross-validation with bootstrap confidence intervals (1,000 iterations):

| Region | n_train | AUC | 95% CI |
|--------|---------|-----|--------|
| Hand | 1,510 | 0.850 | [0.829, 0.873] |
| Leg | 2,237 | 0.888 | [0.859, 0.914] |
| Hip | 179 | 0.864 | [0.764, 0.953]* |
| Shoulder | 98 | 0.848 | [0.664, 0.972]* |
| **Overall** | **4,024** | **0.882** | -- |

\* Wide CI due to small sample size — statistically unreliable.

### Data Efficiency

The probe achieves strong performance with remarkably few labeled examples, demonstrating that CXR Foundation embeddings carry transferable signal even for out-of-domain musculoskeletal anatomy:

| Training examples | AUC |
|-------------------|-----|
| 10 | 0.555 |
| 25 | 0.578 |
| 50 | 0.607 |
| 100 | 0.683 |
| 250 | 0.785 |
| 500 | 0.820 |
| 4,024 (full) | 0.882 |

0.820 AUC at just 500 examples suggests viable deployment in data-scarce clinical settings.

## Usage

```python
import joblib
import numpy as np

# Load the probe
probe = joblib.load("fracture_probe.joblib")
model = probe["model"]
scaler = probe["scaler"]
temperature = probe["temperature"]

# Given a CXR Foundation embedding vector
# embedding = extract_embedding(image)  # shape: (88064,)

embedding_scaled = scaler.transform(embedding.reshape(1, -1))
logit = model.decision_function(embedding_scaled)
probability = 1 / (1 + np.exp(-logit / temperature))
```

## Intended Use

- **Primary:** Fracture screening triage in resource-limited settings without radiologist coverage
- **Deployment target:** NVIDIA Jetson Orin Nano (8 GB) or similar edge devices
- **NOT intended for:** Standalone clinical diagnosis. This is a screening aid — all positive findings require radiologist confirmation.

## Limitations

- Trained only on FracAtlas (hand, leg, hip, shoulder). Performance on other body regions is unknown.
- Hip and shoulder results have wide confidence intervals due to small sample sizes (179 and 98 training examples respectively).
- Requires CXR Foundation for embedding extraction, which expects chest X-ray-format input images.
- Not validated in a prospective clinical trial.

## Integrity

SHA-256 checksum for `fracture_probe.joblib`:
```
5ab04c187564f84cd26dfac5467f2949094f1045efc9b59d2d97260ec3f0dad6
```

A `.sha256` sidecar file is included in this repository for automated verification.

## Citation

If you use this model, please cite:

```
@misc{edgefracture2026,
  title={EdgeFracture: CXR Foundation Fracture Probe for Edge Deployment},
  author={Ibrahim Dagib},
  year={2026},
  url={https://huggingface.co/ibdagib/edgefracture-cxr-fracture-probe}
}
```

## Part of EdgeFracture

This probe is one component of the [EdgeFracture](https://github.com/ibdagib/edgefracture) pipeline — a two-model fracture triage system combining CXR Foundation embeddings with MedGemma clinical reasoning, built for the Google MedGemma Impact Challenge.
