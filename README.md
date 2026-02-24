# EdgeFracture — Portable Musculoskeletal Fracture Triage on Edge Hardware

A portable, fully offline fracture screening station that repurposes a chest X-ray foundation model (`google/cxr-foundation`) for musculoskeletal fracture triage on a $249 NVIDIA Jetson Orin Nano.

## The Problem

Missed fractures are the #1 diagnostic error in emergency departments (estimated 3-10% miss rate).  
4.3 billion people live in regions with fewer than 1 radiologist per 100,000 population.

## Our Approach

Google's CXR Foundation was trained only on chest X-rays, but transfers to musculoskeletal fracture screening across 4 FracAtlas regions (Hand, Leg, Hip, Shoulder).  
With as few as 500 labeled examples, the linear probe reaches 0.82 AUC.  
MedGemma 1.5 4B adds body-region detection, visual cross-checking, and clinician/patient communication.

## HAI-DEF Models Used

- **CXR Foundation** (`google/cxr-foundation`) — image embeddings + fracture classification
- **MedGemma 1.5 4B** (`google/medgemma-1.5-4b-it`) — body region detection, visual safety audit, and report generation

## How This Result Is Produced (App Pipeline)

1. **Step 0 - Region Detection:** MedGemma vision analyzes the uploaded X-ray and predicts the anatomical region (for example, `Hand`, `Leg`, `Hip`, `Shoulder`, `Foot`).
2. **Step 1 - Image Analysis:** CXR Foundation processes the image and generates a high-dimensional embedding (88,064 features in the current app pipeline).
3. **Step 2 - Fracture Screening:** A temperature-calibrated logistic regression linear probe scores fracture likelihood from the embedding.
4. **Step 3 - Clinical Report + Cross-Check:** MedGemma generates clinician and patient summaries and runs a visual safety audit to assess concordance with the CXR-based score.

## What the Triage Score Means

- **>= 70%**: `HIGH SUSPICION` (red) - prioritize radiologist review.
- **40-69.9%**: `MODERATE SUSPICION` (yellow) - uncertain range; correlate clinically and consider further imaging.
- **< 40%**: `LOW SUSPICION` (green) - lower likelihood, but subtle/non-displaced fractures can still be missed.

## Important Limitations

- CXR Foundation was trained on chest X-rays only; musculoskeletal use is transfer learning.
- This is a screening tool, not a diagnostic device.
- Performance varies by body region and fracture type.
- All outputs require review by a qualified healthcare professional.
- The linear probe is formally validated on FracAtlas regions only (Hand, Leg, Hip, Shoulder). Other regions (for example, Foot) should be interpreted with extra caution.

## Model Performance Snapshot

- **Overall AUC:** 0.882 (5-fold CV) on FracAtlas (4,024 evaluated studies, 717 fractures).
- **Per-region AUCs:** Hand 0.850, Leg 0.888, Hip 0.864, Shoulder 0.848.
- **Data efficiency:** 0.820 AUC with 500 labeled training examples.

## Quick Start

### 1. Environment Setup (Jetson Orin Nano)

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/edgefracture.git
cd edgefracture

# Create virtual environment
python3 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama for MedGemma
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M
```

### 2. Download Models & Data

```bash
# Download FracAtlas dataset
python scripts/download_fracatlas.py

# Download CXR Foundation model
python scripts/download_cxr_foundation.py
```

### 3. Run Experiments

```bash
# Extract embeddings from all FracAtlas images
python scripts/01_extract_embeddings.py

# Run zero-shot classification (Experiment 1)
python scripts/02_zero_shot.py

# Train linear probe + data efficiency curve (Experiment 2)
python scripts/03_linear_probe.py

# Run edge benchmarks (Experiment 3)
python scripts/04_edge_benchmarks.py
```

### 4. Launch Demo App

```bash
python app/app.py
# Open http://localhost:7860
```

After upload, the app runs triage, visual cross-check, and report generation in sequence, then exposes an Evidence panel with transparency and benchmark context.

## Hardware

| Component | Spec |
|-----------|------|
| Device | NVIDIA Jetson Orin Nano 8GB (Super) |
| GPU | 1024-core Ampere, 67 TOPS |
| Memory | 8GB LPDDR5 unified |
| Cost | $249 USD |

## Dataset

- **FracAtlas** — 4,083 musculoskeletal X-rays (717 fracture cases) across Hand, Leg, Hip, Shoulder
- License: CC BY 4.0

## Competition

MedGemma Impact Challenge 2026 (Kaggle / Google Health AI)  
Target: Edge AI Prize + Novel Task Prize + Main Track
