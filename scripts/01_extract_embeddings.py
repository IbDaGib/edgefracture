#!/usr/bin/env python3
"""
Experiment 0: Extract CXR Foundation embeddings from all FracAtlas images.

This is the foundation for all subsequent experiments.
CXR Foundation outputs ELIXR-contrastive embeddings (language-aligned).

Two embedding types:
  - Contrastive (ELIXR-C): 32x128 image embeddings, aligned with text
  - v2.0 text: Text embeddings from the BERT encoder

Usage:
    python scripts/01_extract_embeddings.py [--device cuda] [--batch-size 4]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path("models/cxr-foundation")
DATA_DIR = Path("data/fracatlas")
OUTPUT_DIR = Path("results/embeddings")

# CXR Foundation expects 1280x1280 images
IMAGE_SIZE = 1280

# Preprocessing: match CXR Foundation training pipeline
# The model expects single-channel (grayscale) images normalized to [0, 1]
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
])


# ============================================================
# Model Loading
# ============================================================

def load_cxr_foundation(model_dir: Path, device: str):
    """
    Load the CXR Foundation model.
    
    CXR Foundation has two components:
    1. Vision encoder (EfficientNet-L2) → image embeddings
    2. Text encoder (BERT) → text embeddings
    
    These are aligned via ELIXR contrastive training on 821K chest X-rays.
    
    NOTE: The exact loading mechanism depends on how Google packaged the model.
    This script tries multiple approaches. You may need to adapt based on
    the actual model files you downloaded.
    """
    print(f"Loading CXR Foundation from {model_dir}...")
    print(f"Device: {device}")
    
    # List available files
    model_files = list(model_dir.rglob("*"))
    print(f"Model files found: {[str(f.relative_to(model_dir)) for f in model_files if f.is_file()]}")
    
    # Approach 1: Try loading via HuggingFace transformers (if supported)
    try:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        model = model.to(device).eval()
        print("Loaded via HuggingFace AutoModel")
        return {"type": "huggingface", "model": model, "processor": processor}
    except Exception as e:
        print(f"HuggingFace AutoModel failed: {e}")
    
    # Approach 2: Try loading TF SavedModel converted to PyTorch
    # CXR Foundation was originally a TensorFlow model
    try:
        # Check for .safetensors or .bin files
        safetensor_files = list(model_dir.glob("*.safetensors"))
        bin_files = list(model_dir.glob("*.bin"))
        
        if safetensor_files or bin_files:
            from safetensors.torch import load_file
            if safetensor_files:
                state_dict = load_file(str(safetensor_files[0]))
                print(f"Loaded state dict from {safetensor_files[0].name}")
                print(f"Keys: {list(state_dict.keys())[:10]}...")
                return {"type": "state_dict", "state_dict": state_dict}
    except Exception as e:
        print(f"Safetensors loading failed: {e}")
    
    # Approach 3: Direct TF/JAX loading (if on Jetson with TF)
    try:
        import tensorflow as tf
        saved_model_dirs = [d for d in model_dir.iterdir() if d.is_dir() and (d / "saved_model.pb").exists()]
        if saved_model_dirs:
            tf_model = tf.saved_model.load(str(saved_model_dirs[0]))
            print(f"Loaded TF SavedModel from {saved_model_dirs[0]}")
            return {"type": "tensorflow", "model": tf_model}
    except Exception as e:
        print(f"TensorFlow loading failed: {e}")
    
    # Approach 4: Try the Google-provided inference utilities
    # Check if there's a custom inference script in the model directory
    try:
        sys.path.insert(0, str(model_dir))
        # Some Google models include their own loading code
        import importlib
        for py_file in model_dir.glob("*.py"):
            mod = importlib.import_module(py_file.stem)
            if hasattr(mod, "load_model") or hasattr(mod, "CXRFoundation"):
                print(f"Found custom loader in {py_file.name}")
                return {"type": "custom", "module": mod}
    except Exception as e:
        pass
    
    print("\n" + "="*60)
    print("ERROR: Could not auto-load CXR Foundation.")
    print("="*60)
    print("\nPlease check the model files and update the loading code.")
    print("You can also try running the Google Colab notebook first")
    print("to understand the exact loading mechanism, then adapt this script.")
    print("\nAlternative: Use the HuggingFace reference demo code as a template:")
    print("  https://huggingface.co/google/cxr-foundation")
    print("\nAs a fallback, you can extract embeddings on Kaggle/Colab")
    print("and copy the .npy files to the Jetson for the rest of the pipeline.")
    sys.exit(1)


def extract_image_embedding(model_info: dict, image: torch.Tensor, device: str):
    """
    Extract ELIXR-contrastive embedding from a single image.
    Returns numpy array of shape (embedding_dim,) — flattened from 32x128.
    """
    with torch.no_grad():
        if model_info["type"] == "huggingface":
            model = model_info["model"]
            # Try standard forward pass
            inputs = image.unsqueeze(0).to(device)
            outputs = model(inputs)
            # The exact output key depends on the model
            if hasattr(outputs, "image_embeds"):
                emb = outputs.image_embeds
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1)
            elif isinstance(outputs, dict):
                # Try common keys
                for key in ["image_embedding", "embeddings", "image_embeds", "logits"]:
                    if key in outputs:
                        emb = outputs[key]
                        break
                else:
                    emb = list(outputs.values())[0]
            else:
                emb = outputs
            
            return emb.cpu().numpy().flatten()
        
        elif model_info["type"] == "tensorflow":
            import tensorflow as tf
            tf_model = model_info["model"]
            img_np = image.numpy()
            # TF models often expect different input format
            tf_input = tf.constant(img_np[np.newaxis, ...])
            output = tf_model(tf_input)
            if isinstance(output, dict):
                emb = list(output.values())[0].numpy()
            else:
                emb = output.numpy()
            return emb.flatten()
        
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")


def extract_text_embedding(model_info: dict, text: str, device: str):
    """
    Extract text embedding from the BERT text encoder.
    Used for zero-shot classification.
    """
    with torch.no_grad():
        if model_info["type"] == "huggingface":
            processor = model_info.get("processor")
            model = model_info["model"]
            
            if processor:
                inputs = processor(text=text, return_tensors="pt").to(device)
                outputs = model.get_text_features(**inputs)
            else:
                # Try tokenizer approach
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
                inputs = tokenizer(text, return_tensors="pt").to(device)
                outputs = model.encode_text(inputs)
            
            return outputs.cpu().numpy().flatten()
    
    raise NotImplementedError("Text embedding extraction not implemented for this model type")


# ============================================================
# Main Extraction Pipeline
# ============================================================

def find_images(data_dir: Path) -> list:
    """Find all X-ray images in the FracAtlas dataset."""
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = []
    
    for ext in image_extensions:
        images.extend(data_dir.rglob(f"*{ext}"))
        images.extend(data_dir.rglob(f"*{ext.upper()}"))
    
    # Deduplicate
    images = sorted(set(images))
    print(f"Found {len(images)} images in {data_dir}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Extract CXR Foundation embeddings")
    # Auto-detect best device: CUDA (Jetson/GPU) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (1 for Jetson)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_info = load_cxr_foundation(args.model_dir, args.device)
    
    # Find images
    images = find_images(args.data_dir)
    if not images:
        print("No images found. Run scripts/download_fracatlas.py first.")
        sys.exit(1)
    
    # Load labels if available
    labels_path = args.data_dir / "labels.csv"
    labels_df = None
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        print(f"Loaded labels: {len(labels_df)} entries")
    
    # Extract embeddings
    embeddings = []
    metadata = []
    timings = []
    
    print(f"\nExtracting embeddings from {len(images)} images...")
    
    for img_path in tqdm(images, desc="Extracting"):
        try:
            # Load and preprocess
            img = Image.open(img_path).convert("L")  # Grayscale
            img_tensor = TRANSFORM(img)
            
            # Extract
            t0 = time.time()
            emb = extract_image_embedding(model_info, img_tensor, args.device)
            t1 = time.time()
            
            embeddings.append(emb)
            timings.append(t1 - t0)
            
            # Match with labels
            fname = img_path.name
            meta = {"file_name": fname, "file_path": str(img_path)}
            
            if labels_df is not None:
                match = labels_df[labels_df["file_name"].str.contains(fname, na=False)]
                if len(match) > 0:
                    meta["has_fracture"] = int(match.iloc[0]["has_fracture"])
                    meta["body_region"] = match.iloc[0].get("body_region", "unknown")
                else:
                    meta["has_fracture"] = -1  # Unknown
                    meta["body_region"] = "unknown"
            
            metadata.append(meta)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Save results
    embeddings_array = np.array(embeddings)
    np.save(args.output_dir / "embeddings.npy", embeddings_array)
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(args.output_dir / "metadata.csv", index=False)
    
    # Save timing stats
    timing_stats = {
        "total_images": len(embeddings),
        "embedding_dim": embeddings_array.shape[1] if len(embeddings) > 0 else 0,
        "mean_latency_ms": np.mean(timings) * 1000,
        "median_latency_ms": np.median(timings) * 1000,
        "p95_latency_ms": np.percentile(timings, 95) * 1000,
        "total_time_s": sum(timings),
    }
    with open(args.output_dir / "extraction_stats.json", "w") as f:
        json.dump(timing_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Embeddings saved: {args.output_dir / 'embeddings.npy'}")
    print(f"Shape: {embeddings_array.shape}")
    print(f"Metadata saved: {args.output_dir / 'metadata.csv'}")
    print(f"Mean extraction latency: {timing_stats['mean_latency_ms']:.1f} ms")
    print(f"Total time: {timing_stats['total_time_s']:.1f} s")
    print(f"{'='*60}")
    print(f"\nNext step: python scripts/02_zero_shot.py")


if __name__ == "__main__":
    main()
