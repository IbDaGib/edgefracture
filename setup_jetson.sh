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

# 2. Virtual environment
echo "[2/6] Creating Python environment..."
python3 -m venv venv
source venv/bin/activate

# 3. PyTorch for Jetson (JetPack-compatible wheels)
echo "[3/6] Installing PyTorch for Jetson..."
# JetPack 6.x comes with PyTorch support. If not pre-installed:
pip install --upgrade pip
# For Jetson Orin Nano with JetPack 6.x, use the NVIDIA-provided wheel:
# Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# pip install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/
pip install torch torchvision 2>/dev/null || echo "NOTE: Install PyTorch from NVIDIA Jetson wheels if this fails"

# 4. Python dependencies
echo "[4/6] Installing Python packages..."
pip install -r requirements.txt

# 5. Ollama for MedGemma
echo "[5/6] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "[5/6] Pulling MedGemma model..."
# Try the Q4 quantized version first (fits in 8GB)
ollama pull hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M 2>/dev/null || \
ollama pull medgemma 2>/dev/null || \
echo "WARNING: Could not pull MedGemma. You may need to manually download a GGUF."

# 6. HuggingFace login
echo "[6/6] HuggingFace authentication..."
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
echo "  source venv/bin/activate"
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
