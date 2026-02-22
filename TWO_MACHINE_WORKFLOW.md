# EdgeFracture — Two-Machine Workflow

## The Setup

| Machine | Role | What Runs Here |
|---------|------|----------------|
| **MacBook Pro M4** | Development + experiments | CXR Foundation embedding extraction, linear probe training, zero-shot experiments, all plotting, code iteration |
| **Jetson Orin Nano** | Deployment + demo | Gradio app, MedGemma via Ollama, edge benchmarks, video recording |

This is the optimal approach. The Mac is faster, has more RAM, and lets you
iterate quickly. The Jetson is for the final demo and edge benchmarks only.

---

## What Goes Where

### On Mac (do ALL of this first):
1. Download FracAtlas dataset
2. Download CXR Foundation model
3. Extract all embeddings → `embeddings.npy`
4. Run zero-shot experiment → results + plots
5. Train linear probe + data efficiency curve → `fracture_probe.joblib` + plots
6. Test MedGemma via Ollama (Mac runs it great)
7. Test Gradio app locally to make sure it works

### Transfer to Jetson (just these files):
```
embeddings.npy          (~50-150 MB)  — only if you want to re-run experiments
metadata.csv            (~200 KB)
fracture_probe.joblib   (~1 KB)       — the trained linear probe
data_efficiency_curve.png              — for the writeup
combined_summary.png                   — for the writeup
per_region_auc.png                     — for the writeup
app/app.py                             — the Gradio demo
```

### On Jetson (demo + benchmarks only):
1. Load CXR Foundation (or use pre-extracted embeddings if it won't load)
2. Run MedGemma via Ollama
3. Run Gradio app
4. Capture edge benchmarks (latency, memory, power)
5. Record the demo video showing the physical hardware

---

## Mac Setup (Step by Step)

### 1. Environment

```bash
# Clone/extract the project
cd ~/projects
tar -xzf edgefracture.tar.gz
cd edgefracture

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch (MPS = Apple Silicon GPU)
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
# MPS is Apple's GPU backend — CXR Foundation will use this
"
```

### 2. HuggingFace Login

```bash
pip install huggingface-hub
huggingface-cli login
# Paste your token (get one at https://huggingface.co/settings/tokens)

# IMPORTANT: Accept the CXR Foundation license first:
# https://huggingface.co/google/cxr-foundation
# Click "Agree and access repository"
```

### 3. Download Everything

```bash
# FracAtlas dataset
python scripts/download_fracatlas.py

# CXR Foundation model
python scripts/download_cxr_foundation.py
```

### 4. Run Experiments

```bash
# Use MPS (Apple GPU) or CPU — both work on M4
# MPS is faster but some ops may fall back to CPU

# Extract embeddings (the big one — may take 30-60 min)
python scripts/01_extract_embeddings.py --device mps
# If MPS gives errors, fall back to CPU:
# python scripts/01_extract_embeddings.py --device cpu

# Zero-shot experiment
python scripts/02_zero_shot.py

# Linear probe + data efficiency (this is fast, < 1 min)
python scripts/03_linear_probe.py

# Check your results!
open results/linear_probe/data_efficiency_curve.png
open results/linear_probe/combined_summary.png
```

### 5. Test the App Locally

```bash
# Install and start Ollama on Mac
brew install ollama
ollama serve &

# Pull MedGemma
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M

# Launch the app
python app/app.py

# Open http://localhost:7860
# Test with a few FracAtlas images
```

### 6. Transfer to Jetson

```bash
# Find your Jetson's IP
# On Jetson: hostname -I

# Create a transfer bundle
mkdir -p transfer_to_jetson/results/linear_probe
mkdir -p transfer_to_jetson/results/embeddings
mkdir -p transfer_to_jetson/results/zero_shot

# Copy what the Jetson needs
cp results/linear_probe/fracture_probe.joblib transfer_to_jetson/results/linear_probe/
cp results/linear_probe/*.png transfer_to_jetson/results/linear_probe/
cp results/linear_probe/*.json transfer_to_jetson/results/linear_probe/
cp results/embeddings/embeddings.npy transfer_to_jetson/results/embeddings/
cp results/embeddings/metadata.csv transfer_to_jetson/results/embeddings/
cp results/zero_shot/*.json transfer_to_jetson/results/zero_shot/
cp results/zero_shot/*.png transfer_to_jetson/results/zero_shot/
cp -r app/ transfer_to_jetson/app/
cp -r scripts/ transfer_to_jetson/scripts/
cp requirements.txt transfer_to_jetson/
cp README.md transfer_to_jetson/

# Also grab some test images for the demo
mkdir -p transfer_to_jetson/demo_images
# Copy 5-10 representative FracAtlas images (mix of fracture/normal, different regions)
# Pick manually from data/fracatlas/

# Send to Jetson
JETSON_IP="YOUR_JETSON_IP"
rsync -avz --progress transfer_to_jetson/ ${JETSON_IP}:~/edgefracture/

# Or via SCP:
# scp -r transfer_to_jetson/* username@${JETSON_IP}:~/edgefracture/
```

---

## Jetson Setup (Minimal — Just for Demo + Benchmarks)

### 1. Basics

```bash
# SSH in from your Mac
ssh username@JETSON_IP

# Add swap (important for 8GB)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Install jtop for monitoring
sudo pip3 install jetson-stats
```

### 2. PyTorch on Jetson

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Get the right PyTorch wheel from:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Install it (example — use YOUR JetPack version's URL):
pip3 install --no-cache-dir torch-2.3.0-cp310-cp310-linux_aarch64.whl

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 3. Python Packages

```bash
cd ~/edgefracture
pip3 install --user \
    transformers huggingface-hub safetensors \
    numpy pandas scikit-learn scipy Pillow \
    matplotlib gradio tqdm psutil ollama
```

### 4. CXR Foundation on Jetson

```bash
# Option A: Try loading it directly
python3 scripts/download_cxr_foundation.py
python3 -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('models/cxr-foundation', trust_remote_code=True)
print('CXR Foundation loaded on Jetson ✓')
"

# Option B (if it fails): That's fine!
# You already have embeddings.npy from the Mac.
# The Gradio app can use the linear probe directly.
# For the LIVE DEMO, we still show CXR Foundation inference on Jetson
# if possible — but the experiments are already done.
```

### 5. MedGemma on Jetson

```bash
# Start Ollama
sudo systemctl start ollama

# Pull MedGemma Q4
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M

# Test it
ollama run medgemma "What is a Colles fracture? Two sentences."

# Check memory usage while running
jtop  # in another terminal
```

### 6. Launch Demo

```bash
cd ~/edgefracture
python3 app/app.py --port 7860

# Access from your Mac's browser:
# http://JETSON_IP:7860
```

### 7. Edge Benchmarks

```bash
# Run with jtop open in another terminal to watch
python3 scripts/04_edge_benchmarks.py --n-images 20 --n-prompts 3
```

### 8. Record Video

The video needs to show the physical Jetson. Options:
- **Phone camera** pointed at the Jetson + screen showing the app
- **Screen recording** on your Mac (accessing the Jetson's Gradio via browser)
  combined with a photo/clip of the Jetson hardware
- Best approach: split-screen — Jetson hardware on one side, app UI on the other

---

## Timeline with Two Machines

### TODAY — Mac (all experiments)
| Time | Task | Machine |
|------|------|---------|
| 1 hr | Environment setup, downloads | Mac |
| 2 hr | CXR Foundation loading + embedding extraction | Mac |
| 30 min | Zero-shot experiment | Mac |
| 30 min | Linear probe + data efficiency | Mac |
| 30 min | Test Gradio app locally | Mac |
| 30 min | Review results, pick demo images | Mac |

### TONIGHT / TOMORROW MORNING — Jetson setup
| Time | Task | Machine |
|------|------|---------|
| 1 hr | Jetson basics (PyTorch, swap, packages) | Jetson |
| 30 min | Transfer files from Mac | Both |
| 30 min | Ollama + MedGemma setup | Jetson |
| 30 min | Test Gradio app on Jetson | Jetson |

### TOMORROW — Demo + Submission
| Time | Task | Machine |
|------|------|---------|
| 1 hr | Edge benchmarks | Jetson |
| 1.5 hr | Record video | Both |
| 2 hr | Write 3-page writeup | Mac |
| 1 hr | Clean GitHub repo, submit | Mac |
