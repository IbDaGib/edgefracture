# EdgeFracture — Two-Machine Workflow

## The Setup

| Machine | Role | What Runs Here |
|---------|------|----------------|
| **MacBook Pro M4** | Development + experiments | CXR Foundation embedding extraction, linear probe training, zero-shot experiments, all plotting, code iteration |
| **Jetson Orin Nano 8GB** | Deployment + demo | Gradio app, MedGemma via Ollama, edge benchmarks, video recording |

This is the optimal approach. The Mac is faster, has more RAM, and lets you
iterate quickly. The Jetson is for the final demo and edge benchmarks only.

---

## Current Status

**Mac-side work is complete.** All experiments have been run and artifacts exist:

| Artifact | Path | Size |
|----------|------|------|
| CXR Foundation model | `models/cxr-foundation/` | 2.6 GB |
| Pooled embeddings | `results/embeddings/embeddings.npy` | 1.3 GB |
| Contrastive embeddings | `results/embeddings/contrastive_embeddings.npy` | 63 MB |
| Fracture probe | `results/linear_probe/fracture_probe.joblib` | 2.7 MB |
| Region classifier | `results/linear_probe/region_classifier.joblib` | 4.7 MB |
| Data efficiency curve | `results/linear_probe/data_efficiency_curve.png` | -- |
| Per-region AUC chart | `results/linear_probe/per_region_auc.png` | -- |
| Combined summary | `results/linear_probe/combined_summary.png` | -- |
| Linear probe results | `results/linear_probe/linear_probe_results.json` | -- |
| Zero-shot results | `results/zero_shot/zero_shot_results.json` | -- |
| FracAtlas dataset | `data/fracatlas/` | 675 MB |

**What remains: Jetson deployment, benchmarks, video, and writeup.**

---

## What Goes Where

### On Mac (DONE):
1. ~~Download FracAtlas dataset~~
2. ~~Download CXR Foundation model~~
3. ~~Extract all embeddings~~
4. ~~Run zero-shot experiment~~
5. ~~Train linear probe + data efficiency curve~~
6. ~~Test MedGemma via Ollama~~
7. ~~Test Gradio app locally~~

### Transfer to Jetson (these files):

**Minimum for demo (app only, ~2.7 GB):**
```
app/                                   — Gradio app (all modules)
models/cxr-foundation/                 — CXR Foundation TF SavedModel (2.6 GB)
results/linear_probe/fracture_probe.joblib     — trained linear probe (2.7 MB)
results/linear_probe/region_classifier.joblib  — region classifier (4.7 MB)
requirements.txt
setup_jetson.sh
```

**Full transfer (for re-running experiments + writeup charts, ~4.1 GB):**
```
app/                                   — Gradio app
scripts/                               — all experiment scripts
models/cxr-foundation/                 — CXR Foundation (2.6 GB)
results/                               — all results, charts, probes (1.4 GB)
requirements.txt
setup_jetson.sh
README.md
```

**If CXR Foundation won't load on Jetson (fallback, ~1.4 GB):**
```
app/                                   — Gradio app
results/embeddings/embeddings.npy      — pre-extracted embeddings (1.3 GB)
results/embeddings/metadata.csv
results/linear_probe/                  — probes + charts
requirements.txt
```

### On Jetson (remaining work):
1. Run `setup_jetson.sh` (installs PyTorch, TF, Ollama, swap)
2. Load CXR Foundation (or use pre-extracted embeddings if it won't load)
3. Pull MedGemma via Ollama
4. Launch Gradio app, test end-to-end
5. Run edge benchmarks (latency, memory, power)
6. Record demo video showing physical hardware

---

## Transfer to Jetson

```bash
# Find your Jetson's IP
# On Jetson: hostname -I
JETSON_IP="YOUR_JETSON_IP"
JETSON_USER="YOUR_USERNAME"

# Option A: rsync the whole project (recommended, ~4.1 GB)
# Excludes raw data zip, venvs, caches, and git
rsync -avz --progress \
    --exclude='venv*' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.cache' \
    --exclude='data/fracatlas/raw/FracAtlas/images/' \
    --exclude='data/fracatlas/fracatlas_raw.zip' \
    --exclude='models/cxr-foundation/.cache/' \
    ./ ${JETSON_USER}@${JETSON_IP}:~/edgefracture/

# Option B: Minimal transfer (just what the app needs)
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ~/edgefracture/{app,models,results/linear_probe,results/embeddings}"

rsync -avz --progress app/ ${JETSON_USER}@${JETSON_IP}:~/edgefracture/app/
rsync -avz --progress scripts/ ${JETSON_USER}@${JETSON_IP}:~/edgefracture/scripts/
rsync -avz --progress models/cxr-foundation/ ${JETSON_USER}@${JETSON_IP}:~/edgefracture/models/cxr-foundation/ \
    --exclude='.cache' --exclude='precomputed_embeddings'
rsync -avz --progress results/linear_probe/ ${JETSON_USER}@${JETSON_IP}:~/edgefracture/results/linear_probe/
rsync -avz --progress results/embeddings/ ${JETSON_USER}@${JETSON_IP}:~/edgefracture/results/embeddings/

scp requirements.txt setup_jetson.sh README.md ${JETSON_USER}@${JETSON_IP}:~/edgefracture/

# Also grab some test images for the demo
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ~/edgefracture/demo_images"
# Copy 5-10 representative FracAtlas images (mix of fracture/normal, different regions)
# Pick manually from data/fracatlas/raw/FracAtlas/images/
```

---

## Jetson Setup (Minimal — Just for Demo + Benchmarks)

### 1. Initial Setup

```bash
# SSH in from your Mac
ssh ${JETSON_USER}@${JETSON_IP}
cd ~/edgefracture

# Run the automated setup script (installs PyTorch, TF, Ollama, swap, packages)
bash setup_jetson.sh

# Activate the environment
source venv311/bin/activate
```

If `setup_jetson.sh` has issues, see `JETSON_QUICKSTART.md` for step-by-step manual setup.

### 2. Set Performance Mode

```bash
# Max performance (MAXN)
sudo nvpmodel -m 0
sudo jetson_clocks

# JetPack 6.2+: check if MAXN SUPER is available (up to 2x gen AI perf)
sudo nvpmodel --listmodes
# If listed, use it for benchmarks:
# sudo nvpmodel -m <super_mode_id>
```

### 3. Verify Environment

```bash
# PyTorch + CUDA
python3 -c "import torch; assert torch.cuda.is_available(); print('PyTorch CUDA OK')"

# TensorFlow + GPU
python3 -c "import tensorflow as tf; print(f'TF GPUs: {tf.config.list_physical_devices(\"GPU\")}')"

# Ollama
ollama list
```

### 4. CXR Foundation on Jetson

```bash
# Option A: Load the TF SavedModel directly
python3 -c "
import tensorflow as tf
model = tf.saved_model.load('models/cxr-foundation/elixr-c-v2-pooled')
print('CXR Foundation loaded on Jetson')
print(f'Signatures: {list(model.signatures.keys())}')
"

# Option B (if TF won't load or OOMs): That's fine!
# You already have embeddings.npy from the Mac.
# The Gradio app can use the linear probe directly with pre-extracted embeddings.
# For the LIVE DEMO, we still show CXR Foundation inference on Jetson
# if possible — but the experiments are already done.
```

### 5. MedGemma on Jetson

```bash
# Start Ollama
sudo systemctl start ollama

# Check what model was pulled during setup
ollama list

# If MedGemma isn't listed, pull it now:
ollama pull alibayram/medgemma
# Fallback: ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M

# Test it
ollama run medgemma "What is a Colles fracture? Two sentences."

# IMPORTANT: If the model name differs from app/config.py default, set it:
export MEDGEMMA_MODEL="alibayram/medgemma"  # match whatever ollama list shows

# Check memory usage while running
jtop  # in another terminal
```

### 6. Launch Demo

```bash
cd ~/edgefracture
source venv311/bin/activate

# Set model name if needed (check ollama list for exact name)
# export MEDGEMMA_MODEL="alibayram/medgemma"

python3 app/app.py --port 7860

# Access from your Mac's browser:
# http://JETSON_IP:7860
```

### 7. Edge Benchmarks

```bash
# Run with jtop open in another terminal to watch
python3 scripts/04_edge_benchmarks.py --n-images 20 --n-prompts 3

# Results saved to results/edge_benchmarks/
```

### 8. Record Video

The video needs to show the physical Jetson. Options:
- **Phone camera** pointed at the Jetson + screen showing the app
- **Screen recording** on your Mac (accessing the Jetson's Gradio via browser)
  combined with a photo/clip of the Jetson hardware
- Best approach: split-screen — Jetson hardware on one side, app UI on the other

Script it first:
1. Show the Jetson hardware (~10s)
2. Boot + launch the app (~20s)
3. Upload an image, get instant triage result (~30s)
4. Show MedGemma report generation (~30s)
5. Show the data efficiency curve chart (~15s)
6. Show benchmark metrics (~15s)
7. Outro with key numbers (~10s)

---

## Remaining Timeline

### Jetson Setup (~2.5 hours)
| Time | Task | Machine |
|------|------|---------|
| 30 min | Transfer files from Mac | Both |
| 1 hr | Run `setup_jetson.sh`, verify environment | Jetson |
| 30 min | Ollama + MedGemma setup + test | Jetson |
| 30 min | Test Gradio app on Jetson | Jetson |

### Demo + Submission (~5.5 hours)
| Time | Task | Machine |
|------|------|---------|
| 1 hr | Edge benchmarks | Jetson |
| 1.5 hr | Record video | Both |
| 2 hr | Write 3-page writeup | Mac |
| 1 hr | Clean GitHub repo, submit | Mac |
