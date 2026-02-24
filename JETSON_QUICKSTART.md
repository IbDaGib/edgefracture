# Jetson Orin Nano — First Project Quickstart

Everything you need to know to go from "basic stuff is set up" to running EdgeFracture.

> **Recommended approach:** Do all experiments on your Mac first, then transfer
> trained artifacts to the Jetson for demo + benchmarks only.
> See `TWO_MACHINE_WORKFLOW.md` for the full two-machine deployment guide.

---

## 1. Confirm What You Have

SSH into your Jetson (or open a terminal if you have a monitor attached) and run these:

```bash
# What JetPack version?
cat /etc/nv_tegra_release
# or
dpkg -l | grep nvidia-jetpack

# What Ubuntu version?
lsb_release -a

# How much RAM?
free -h

# GPU info
cat /proc/device-tree/model
```

You should see: Jetson Orin Nano, 8GB RAM, Ubuntu 22.04.

**Expected JetPack versions:**
- **JetPack 6.2.2** (latest as of Feb 2026) — L4T R36.4, CUDA 12.6
- **JetPack 6.2 / 6.2.1** — also fine, same CUDA 12.6 stack

**Write down your JetPack version** — you'll need it for PyTorch.

---

## 2. The Big Thing About Jetson: Unified Memory

Unlike a desktop PC where the GPU has its own separate VRAM, the Jetson shares
ALL 8GB between the CPU and GPU. This means:

- `torch.cuda.is_available()` = True (it has a real NVIDIA GPU)
- But you only have 8GB TOTAL for everything: OS, Python, models, data
- The OS + desktop uses ~1.5-2GB, leaving you ~6GB for models
- This is why we're using quantized models (CXR Foundation ~1GB + MedGemma 1.5 Q4 ~2.5GB = ~3.5GB)

**Pro tip:** If you're running a desktop environment, you can free ~500MB by switching
to text mode: `sudo systemctl set-default multi-user.target` then reboot. Switch back
with `sudo systemctl set-default graphical.target`.

---

## 3. Install the Essentials

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Essential tools
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    htop \
    nano \
    unzip

# jtop — THE essential Jetson monitoring tool (like htop but for GPU/power)
sudo pip3 install jetson-stats
sudo systemctl restart jtop
# Then run it:
jtop
# This shows: GPU usage, memory, power, temperature — all in one place
# Press q to quit
```

---

## 4. PyTorch on Jetson (THIS IS THE TRICKY PART)

**Do NOT just `pip install torch`.** The regular PyPI wheels don't work on Jetson ARM64.
You need Jetson-specific wheels built against your JetPack's CUDA version.

### For JetPack 6.2+ (L4T R36.x, CUDA 12.6):

```bash
# Check your L4T version first
cat /etc/nv_tegra_release
# Look for "R36" — that's JetPack 6

# Option A (recommended): Jetson AI Lab index — PyTorch 2.8.0
pip3 install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Option B: NVIDIA official wheels (check for latest URL)
# https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

**Important:** The Jetson AI Lab wheel requires `numpy==1.26.1`. If you get numpy
version conflicts, pin it: `pip3 install numpy==1.26.1` before installing torch.

### Verify it works:

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Quick test
    x = torch.randn(100, 100, device='cuda')
    print(f'Tensor on GPU: {x.device}')
"
```

**If CUDA shows False:** Your PyTorch wheel doesn't match your JetPack. This is the
#1 issue people hit. Make sure the wheel matches your CUDA version (12.6 for JetPack 6.2+).

---

## 5. TensorFlow on Jetson (Required for CXR Foundation)

CXR Foundation's vision encoder (`elixr-c-v2-pooled`) is a TensorFlow SavedModel.
You need TensorFlow installed alongside PyTorch.

```bash
# For JetPack 6.2+ (CUDA 12.6), install from NVIDIA's index:
pip3 install --no-cache-dir \
    tensorflow==2.16.1+nv24.08 \
    --extra-index-url https://pypi.nvidia.com

# If the NVIDIA wheel isn't available, try the standard wheel:
# pip3 install tensorflow>=2.16.0

# Also need tensorflow-text for the text encoder:
pip3 install tensorflow-text>=2.16.0

# Verify
python3 -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
"
```

**Note:** TensorFlow on ARM64 can be slow to install (may build from source).
If it fails, the fallback is to use pre-extracted embeddings from your Mac —
the app's linear probe only needs numpy/sklearn, not TF directly.

---

## 6. Python Environment

```bash
cd ~/edgefracture

# Use a venv WITH system-site-packages to see Jetson's CUDA libraries
python3 -m venv --system-site-packages venv311
source venv311/bin/activate

# Install project dependencies
pip install -r requirements.txt

# Verify key packages
python3 -c "import torch, transformers, gradio; print('All good')"
```

**Why `--system-site-packages`?** The NVIDIA CUDA libraries and Jetson-specific
PyTorch wheels are installed at system level. A regular venv won't see them.

---

## 7. Install Ollama (for MedGemma)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify it's running
ollama --version
ollama list

# Pull MedGemma — try these options in order:
# Option A: Ollama registry model (simplest)
ollama pull alibayram/medgemma

# Option B: From Unsloth GGUF Q4 quantized (~2.5GB)
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M

# Option C: If neither works, download the GGUF manually:
# wget https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-Q4_K_M.gguf
# echo 'FROM ./medgemma-1.5-4b-it-Q4_K_M.gguf' > Modelfile
# ollama create medgemma -f Modelfile

# Test it
ollama run medgemma "What is a Colles fracture? Answer in 2 sentences."
```

**Note:** If you use a model name different from what's in `app/config.py`, set the
environment variable before launching the app:
```bash
export MEDGEMMA_MODEL="alibayram/medgemma"  # or whatever name ollama list shows
```

**Memory note:** When Ollama loads MedGemma, it uses ~3GB. You can't have both
CXR Foundation AND MedGemma loaded simultaneously in PyTorch. The app handles
this by: (1) running CXR Foundation for fast triage, then (2) calling Ollama
for report generation (Ollama manages its own memory).

---

## 8. File Transfer To/From Jetson

### Option A: SCP (simplest)

```bash
# From your laptop TO the Jetson:
scp edgefracture.tar.gz username@JETSON_IP:~/

# From the Jetson TO your laptop:
scp username@JETSON_IP:~/results/linear_probe/data_efficiency_curve.png ./
```

### Option B: rsync (better for large transfers)

```bash
# Sync project folder to Jetson
rsync -avz --progress ./edgefracture/ username@JETSON_IP:~/edgefracture/

# Sync results back
rsync -avz --progress username@JETSON_IP:~/edgefracture/results/ ./results/
```

### Option C: USB drive

```bash
# On Jetson — USB drives usually auto-mount to /media/username/
ls /media/$(whoami)/
# Copy files
cp -r ~/edgefracture/results/ /media/$(whoami)/USBDRIVE/
```

### Finding your Jetson's IP:

```bash
# On the Jetson:
hostname -I
# or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

---

## 9. Monitoring During Inference

### jtop (recommended — install from step 3)

```bash
jtop
# Shows everything: GPU %, memory, power, temperature, processes
# Tab 1 (ALL): overview
# Tab 2 (GPU): GPU utilization
# Tab 3 (MEM): memory breakdown
# Tab 6 (POWER): watts being used
```

### tegrastats (command line, good for logging)

```bash
# Live monitoring
sudo tegrastats

# Log to file (useful during benchmarks)
sudo tegrastats --interval 500 --logfile tegrastats.log &
# ... run your benchmark ...
kill %1
cat tegrastats.log
```

### nvidia-smi doesn't exist on Jetson
Use `jtop` or `tegrastats` instead. This catches a lot of people off guard.

---

## 10. Common Jetson Gotchas

### "Killed" or OOM errors
You ran out of the 8GB shared memory. Solutions:
- Close the desktop: `sudo systemctl isolate multi-user.target`
- Kill other processes: check `htop`
- Use smaller model quantization (Q4 instead of Q8)
- Process images one at a time, not in batches

### Swap space (VERY HELPFUL for 8GB)
```bash
# Add 4GB swap file — lets you survive temporary memory spikes
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
# Should now show 4GB swap
```

### Thermal throttling
The Jetson will slow down if it overheats. If you notice inference getting
slower over time:
- Check temperature: `cat /sys/class/thermal/thermal_zone*/temp` (divide by 1000 for C)
- Use a fan or heatsink
- Use the higher power mode (see below)
- Set fan to max: `sudo jetson_clocks`

### Power modes
```bash
# Check current mode
sudo nvpmodel -q

# Set to max performance (MAXN)
sudo nvpmodel -m 0

# JetPack 6.2+ Super Mode (Orin Nano 8GB supports MAXN SUPER at 25W):
# Delivers up to 2x gen AI inference performance over standard MAXN.
# Check available modes:
sudo nvpmodel --listmodes
# If MAXN SUPER is listed, use it for benchmarks:
sudo nvpmodel -m <super_mode_id>

# Lock clocks at max (disables dynamic scaling — better for benchmarking)
sudo jetson_clocks
```

### HuggingFace downloads are slow
The Jetson has limited bandwidth processing. For large model downloads:
- Download on your laptop first, then SCP to Jetson
- Or use `huggingface-cli download` which is more robust than Python API

---

## 11. Your First Test Run (Do This NOW)

Run these commands in order. Each one should work before moving to the next.

```bash
# Step 1: Basic GPU test
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE — fix PyTorch first!'
print('GPU works')
x = torch.randn(1, 1, 1280, 1280, device='cuda')
print(f'Can allocate 1280x1280 tensor on GPU')
print(f'Memory used: {torch.cuda.memory_allocated()/1e6:.0f} MB')
"

# Step 2: TensorFlow sees GPU
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow GPUs: {gpus}')
assert len(gpus) > 0, 'TensorFlow cannot see GPU — check TF installation'
print('TensorFlow GPU works')
"

# Step 3: HuggingFace works
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
print('HuggingFace hub works')
"

# Step 4: Ollama works
ollama list
python3 -c "
import ollama
models = ollama.list()
print(f'Ollama models: {[m[\"name\"] for m in models.get(\"models\", [])]}')
print('Ollama works')
"

# Step 5: Gradio works
python3 -c "
import gradio as gr
print(f'Gradio {gr.__version__} works')
"

# Step 6: Memory headroom check
python3 -c "
import psutil
mem = psutil.virtual_memory()
available_gb = mem.available / (1024**3)
print(f'Available RAM: {available_gb:.1f} GB')
if available_gb < 4:
    print('WARNING: Low memory. Close desktop or add swap.')
else:
    print('Memory headroom looks good')
"
```

If all 6 pass, you're ready to start the EdgeFracture pipeline.

---

## Quick Reference

| Task | Command |
|------|---------|
| Check GPU | `jtop` or `cat /proc/device-tree/model` |
| Check memory | `free -h` |
| Check temperature | `cat /sys/class/thermal/thermal_zone*/temp` |
| Max performance | `sudo nvpmodel -m 0 && sudo jetson_clocks` |
| Super Mode (6.2+) | `sudo nvpmodel --listmodes` then set the MAXN SUPER mode |
| Monitor power | `sudo tegrastats` |
| Find IP address | `hostname -I` |
| Start Ollama | `sudo systemctl start ollama` |
| Add swap | `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| Set MedGemma model | `export MEDGEMMA_MODEL="alibayram/medgemma"` |
