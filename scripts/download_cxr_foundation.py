#!/usr/bin/env python3
"""
Download CXR Foundation model from HuggingFace.

Model: google/cxr-foundation
Architecture: EfficientNet-L2 (vision) + BERT (text encoder)
Parameters: ~480M
Output: 32x128 language-aligned embeddings
Training: 821K chest X-rays (no musculoskeletal data)
"""

import os
import sys
from pathlib import Path

MODEL_ID = "google/cxr-foundation"
MODEL_DIR = Path("models/cxr-foundation")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading CXR Foundation from HuggingFace: {MODEL_ID}")
    print("NOTE: This is a gated model. You need to:")
    print("  1. Have a HuggingFace account")
    print("  2. Accept the model license at https://huggingface.co/google/cxr-foundation")
    print("  3. Set your HF token: export HF_TOKEN=your_token_here")
    print("     or: huggingface-cli login")
    print()
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"\nModel downloaded to: {MODEL_DIR}")
        print("Contents:")
        for f in sorted(MODEL_DIR.rglob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.relative_to(MODEL_DIR)} ({size_mb:.1f} MB)")
                
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure you accepted the license on HuggingFace")
        print("  - Set HF_TOKEN environment variable")
        print("  - pip install huggingface-hub")
        sys.exit(1)
    
    print("\nNext step: python scripts/01_extract_embeddings.py")


if __name__ == "__main__":
    main()
