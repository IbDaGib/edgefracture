# EdgeFracture — Portable Musculoskeletal Fracture Triage on Edge Hardware

A portable, fully offline fracture screening station that repurposes a chest X-ray foundation model (`google/cxr-foundation`) to triage musculoskeletal fractures across body regions — running on a $249 NVIDIA Jetson Orin Nano.

## Quick Start

### 1. Environment Setup (Jetson Orin Nano)

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/edgefracture.git
cd edgefracture

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

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
# Open http://localhost:7860 in browser
```

## Hardware

| Component | Spec |
|-----------|------|
| Device | NVIDIA Jetson Orin Nano 8GB (Super) |
| GPU | 1024-core Ampere, 67 TOPS |
| Memory | 8GB LPDDR5 unified |
| Cost | $249 USD |

## Models Used

- **CXR Foundation** (`google/cxr-foundation`) — Zero-shot & linear probe fracture classification
- **MedGemma 1.5 4B** (`medgemma-1.5-4b-it` Q4 GGUF) — Clinical report generation via Ollama

## Dataset

- **FracAtlas** — 4,083 MSK X-rays (717 fracture cases) across Hand, Leg, Hip, Shoulder
- License: CC BY 4.0

## Competition

MedGemma Impact Challenge 2026 — Kaggle / Google Health AI  
Target: Edge AI Prize + Novel Task Prize + Main Track
