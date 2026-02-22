# Jetson Orin Nano — First Project Quickstart

Everything you need to know to go from "basic stuff is set up" to running EdgeFracture.

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

You should see: Jetson Orin Nano, 8GB RAM, Ubuntu 22.04 (JetPack 6.x).

**Write down your JetPack version** — you'll need it for PyTorch.

---

## 2. The Big Thing About Jetson: Unified Memory

Unlike a desktop PC where the GPU has its own separate VRAM, the Jetson shares
ALL 8GB between the CPU and GPU. This means:

- `torch.cuda.is_available()` = True ✓ (it has a real NVIDIA GPU)
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
You need NVIDIA's Jetson-specific wheels.

### For JetPack 6.x (L4T R36.x):

```bash
# Check your L4T version first
cat /etc/nv_tegra_release
# Look for "R36" — that's JetPack 6

# Install PyTorch 2.3+ for JetPack 6
# Go to: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Find the wheel URL for your JetPack version

# Example (URLs change — verify on the forum):
pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl

# Then install torchvision (must match torch version)
pip3 install torchvision --no-build-isolation
```

### For JetPack 5.x (L4T R35.x):

```bash
# Older JetPack — check the NVIDIA forum for matching wheels
# PyTorch 2.0 or 2.1 typically
```

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
    print(f'Tensor on GPU: {x.device} ✓')
"
```

**If CUDA shows False:** Your PyTorch wheel doesn't match your JetPack. This is the
#1 issue people hit. Go back to the NVIDIA forum and get the right wheel.

---

## 5. Python Environment

```bash
# Create project directory
mkdir -p ~/edgefracture
cd ~/edgefracture

# DON'T use a venv for the Jetson PyTorch wheel (it won't see system CUDA)
# Instead, install everything at user level:
pip3 install --user \
    transformers \
    huggingface-hub \
    safetensors \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    Pillow \
    matplotlib \
    seaborn \
    gradio \
    tqdm \
    psutil \
    requests \
    ollama

# Verify key packages
python3 -c "import torch, transformers, gradio; print('All good ✓')"
```

**Note on venvs:** Virtual environments on Jetson can be tricky because the
NVIDIA CUDA libraries are system-level. If you use a venv, you may need
`--system-site-packages` to see CUDA:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

---

## 6. Install Ollama (for MedGemma)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify it's running
ollama --version
ollama list

# Pull MedGemma 1.5 — try Q4 quantized (smallest, ~2.5GB)
# Option A: From Unsloth GGUF (recommended)
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M

# Option B: If that doesn't work, try creating a Modelfile
# First download the GGUF manually:
# wget https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-Q4_K_M.gguf

# Then create a Modelfile:
# echo 'FROM ./medgemma-1.5-4b-it-Q4_K_M.gguf' > Modelfile
# ollama create medgemma -f Modelfile

# Test it
ollama run medgemma "What is a Colles fracture? Answer in 2 sentences."
```

**Memory note:** When Ollama loads MedGemma, it uses ~3GB. You can't have both
CXR Foundation AND MedGemma loaded simultaneously in PyTorch. The app handles
this by: (1) running CXR Foundation for fast triage, then (2) calling Ollama
for report generation (Ollama manages its own memory).

---

## 7. File Transfer To/From Jetson

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

## 8. Monitoring During Inference

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

## 9. Common Jetson Gotchas

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
- Check temperature: `cat /sys/class/thermal/thermal_zone*/temp` (divide by 1000 for °C)
- Use a fan or heatsink
- Use the higher power mode: `sudo nvpmodel -m 0` (MAXN — full performance)
- Set fan to max: `sudo jetson_clocks`

### Power modes
```bash
# Check current mode
sudo nvpmodel -q

# Set to max performance
sudo nvpmodel -m 0

# Lock clocks at max (disables dynamic scaling — better for benchmarking)
sudo jetson_clocks
```

### HuggingFace downloads are slow
The Jetson has limited bandwidth processing. For large model downloads:
- Download on your laptop first, then SCP to Jetson
- Or use `huggingface-cli download` which is more robust than Python API

---

## 10. Your First Test Run (Do This NOW)

Run these commands in order. Each one should work before moving to the next.

```bash
# Step 1: Basic GPU test
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE — fix PyTorch first!'
print('GPU works ✓')
x = torch.randn(1, 1, 1280, 1280, device='cuda')
print(f'Can allocate 1280x1280 tensor on GPU ✓')
print(f'Memory used: {torch.cuda.memory_allocated()/1e6:.0f} MB')
"

# Step 2: HuggingFace works
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
print('HuggingFace hub works ✓')
"

# Step 3: Ollama works
ollama list
python3 -c "
import ollama
models = ollama.list()
print(f'Ollama models: {[m[\"name\"] for m in models.get(\"models\", [])]}')
print('Ollama works ✓')
"

# Step 4: Gradio works
python3 -c "
import gradio as gr
print(f'Gradio {gr.__version__} works ✓')
"

# Step 5: Memory headroom check
python3 -c "
import psutil
mem = psutil.virtual_memory()
available_gb = mem.available / (1024**3)
print(f'Available RAM: {available_gb:.1f} GB')
if available_gb < 4:
    print('⚠️  WARNING: Low memory. Close desktop or add swap.')
else:
    print('Memory headroom looks good ✓')
"
```

If all 5 pass, you're ready to start the EdgeFracture pipeline.

---

## Quick Reference

| Task | Command |
|------|---------|
| Check GPU | `jtop` or `cat /proc/device-tree/model` |
| Check memory | `free -h` |
| Check temperature | `cat /sys/class/thermal/thermal_zone*/temp` |
| Max performance | `sudo nvpmodel -m 0 && sudo jetson_clocks` |
| Monitor power | `sudo tegrastats` |
| Find IP address | `hostname -I` |
| Start Ollama | `sudo systemctl start ollama` |
| Add swap | `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
