#!/bin/bash
# ============================================================
# EdgeFracture — Jetson Orin Nano Quick Setup
# Run this first on a fresh Jetson with JetPack installed
# ============================================================

set -e

echo "=========================================="
echo "EdgeFracture — Jetson Setup"
echo "=========================================="

# 1. System packages
echo "[1/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git curl

# 2. Virtual environment (with system-site-packages to see Jetson's CUDA)
echo "[2/6] Creating Python environment..."
python3 -m venv --system-site-packages venv311
source venv311/bin/activate

# 3. PyTorch + TensorFlow for Jetson (JetPack-compatible wheels)
echo "[3/7] Installing PyTorch for Jetson..."
pip install --upgrade pip
# For JetPack 6.2+ (CUDA 12.6), use the Jetson AI Lab index:
pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126 2>/dev/null || \
    echo "NOTE: PyTorch install failed. Check https://pypi.jetson-ai-lab.io for your JetPack version."

echo "[3/7] Installing TensorFlow for Jetson (needed for CXR Foundation)..."
pip install --no-cache-dir tensorflow>=2.16.0 2>/dev/null || \
    echo "NOTE: TensorFlow install failed. You can still use pre-extracted embeddings."
pip install tensorflow-text>=2.16.0 2>/dev/null || true

# 4. Python dependencies
echo "[4/7] Installing Python packages..."
pip install -r requirements.txt

# 5. Ollama for MedGemma
echo "[5/7] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "[5/7] Pulling MedGemma model..."
# Try Ollama registry first, then Unsloth GGUF fallback
ollama pull alibayram/medgemma 2>/dev/null || \
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M 2>/dev/null || \
echo "WARNING: Could not pull MedGemma. See JETSON_QUICKSTART.md Section 7 for manual options."

# 6. Swap space (important for 8GB unified memory)
echo "[6/7] Adding swap space..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
    echo "4GB swap added."
else
    echo "Swap file already exists, skipping."
fi

# 7. HuggingFace login
echo "[7/7] HuggingFace authentication..."
echo "You need a HuggingFace token to download CXR Foundation (gated model)."
echo "1. Go to https://huggingface.co/settings/tokens"
echo "2. Create a token with 'read' access"
echo "3. Accept the license at https://huggingface.co/google/cxr-foundation"
echo ""
read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN=$HF_TOKEN
    echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
    echo "Token saved."
fi

echo ""
echo "=========================================="
echo "Setup complete! Next steps:"
echo "=========================================="
echo ""
echo "  source venv311/bin/activate"
echo ""
echo "  # Download data and models"
echo "  python scripts/download_fracatlas.py"
echo "  python scripts/download_cxr_foundation.py"
echo ""
echo "  # Run experiments"
echo "  python scripts/01_extract_embeddings.py"
echo "  python scripts/02_zero_shot.py"
echo "  python scripts/03_linear_probe.py"
echo "  python scripts/04_edge_benchmarks.py"
echo ""
echo "  # Launch demo"
echo "  python app/app.py"
echo ""
