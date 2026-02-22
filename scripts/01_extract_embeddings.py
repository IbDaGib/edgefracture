#!/usr/bin/env python3
"""
Extract CXR Foundation embeddings from all FracAtlas images.

CXR Foundation uses TensorFlow SavedModels:
  - elixr-c-v2-pooled: vision encoder → 32x128 contrastive embeddings
  - pax-elixr-b-text: text encoder → 128-dim text embeddings

Usage:
    python scripts/01_extract_embeddings.py
"""

import os
import sys
import io
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tensorflow_text

# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path("models/cxr-foundation")
VISION_MODEL_DIR = MODEL_DIR / "elixr-c-v2-pooled"
TEXT_MODEL_DIR = MODEL_DIR / "pax-elixr-b-text"
LABELS_CSV = Path("data/fracatlas/labels.csv")
RAW_DIR = Path("data/fracatlas/raw")
OUTPUT_DIR = Path("results/embeddings")

IMAGE_SIZE = 1024  # CXR Foundation recommends min 1024x1024


# ============================================================
# Image → tf.Example → Embedding
# ============================================================

def image_to_tf_example(image_path: Path, target_size: int = IMAGE_SIZE) -> bytes:
    """Convert an image to serialized tf.Example with PNG bytes."""
    import tensorflow as tf

    img = Image.open(image_path)
    img = img.convert("L")  # Grayscale for X-rays
    img = img.resize((target_size, target_size), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    feature = {
        "image/encoded": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[png_bytes])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def extract_image_embedding(vision_model, serialized_example: bytes) -> np.ndarray:
    """Extract embedding from a serialized tf.Example."""
    import tensorflow as tf

    input_tensor = tf.constant([serialized_example])

    # Try serving signature first, then direct call
    try:
        if hasattr(vision_model, "signatures") and "serving_default" in vision_model.signatures:
            serve_fn = vision_model.signatures["serving_default"]
            input_keys = list(serve_fn.structured_input_signature[1].keys())
            output = serve_fn(**{input_keys[0]: input_tensor})
        else:
            output = vision_model(input_tensor)
    except Exception as e1:
        try:
            output = vision_model(inputs=input_tensor)
        except Exception as e2:
            raise RuntimeError(
                f"Could not call vision model.\n"
                f"  signatures: {e1}\n"
                f"  direct: {e2}"
            )

    if isinstance(output, dict):
        keys = list(output.keys())
        emb = output[keys[0]].numpy()
        return emb[0]
    else:
        return output.numpy()[0]


def extract_text_embedding(text_model, text: str) -> np.ndarray:
    """Extract text embedding from the BERT text encoder."""
    import tensorflow as tf

    input_tensor = tf.constant([text])

    try:
        if hasattr(text_model, "signatures") and "serving_default" in text_model.signatures:
            serve_fn = text_model.signatures["serving_default"]
            input_keys = list(serve_fn.structured_input_signature[1].keys())
            output = serve_fn(**{input_keys[0]: input_tensor})
        else:
            output = text_model(input_tensor)
    except Exception as e1:
        try:
            output = text_model(inputs=input_tensor)
        except Exception as e2:
            raise RuntimeError(f"Could not call text model: {e1} / {e2}")

    if isinstance(output, dict):
        emb = output[list(output.keys())[0]].numpy()
        return emb[0]
    else:
        return output.numpy()[0]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load labels ----
    labels = pd.read_csv(args.labels_csv)
    print(f"Labels: {len(labels)} images")
    print(f"  Fractures: {labels['has_fracture'].sum()}")
    print(f"  Normal: {(labels['has_fracture'] == 0).sum()}")
    print(f"  Regions: {dict(labels['body_region'].value_counts())}")

    # Build full paths
    labels["full_path"] = labels["file_path"].apply(lambda p: args.raw_dir / p)

    # Verify a few exist
    missing = [r for _, r in labels.iterrows() if not r["full_path"].exists()]
    if missing:
        print(f"WARNING: {len(missing)} images not found!")
        print(f"  Example: {missing[0]['full_path']}")
        labels = labels[labels["full_path"].apply(lambda p: p.exists())].reset_index(drop=True)
        print(f"  Continuing with {len(labels)} images")

    # ---- Load vision model ----
    print("\n" + "=" * 60)
    print("Loading CXR Foundation vision model")
    print("=" * 60)

    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")

    vision_model = tf.saved_model.load(str(VISION_MODEL_DIR))
    print("Vision model loaded ✓")

    if hasattr(vision_model, "signatures"):
        for name, sig in vision_model.signatures.items():
            print(f"  Signature '{name}':")
            print(f"    Inputs:  {sig.structured_input_signature}")
            print(f"    Outputs: {sig.structured_outputs}")

    # ---- Test single image ----
    print("\n" + "=" * 60)
    print("Testing single image")
    print("=" * 60)

    test_path = labels.iloc[0]["full_path"]
    print(f"Image: {test_path}")

    t0 = time.time()
    serialized = image_to_tf_example(test_path, args.image_size)
    t_prep = time.time() - t0

    t0 = time.time()
    emb = extract_image_embedding(vision_model, serialized)
    t_inf = time.time() - t0

    emb_flat = emb.flatten()
    print(f"  Raw shape:  {emb.shape}")
    print(f"  Flat shape: {emb_flat.shape}")
    print(f"  Range:      [{emb.min():.4f}, {emb.max():.4f}]")
    print(f"  Prep time:  {t_prep*1000:.0f} ms")
    print(f"  Infer time: {t_inf*1000:.0f} ms")

    # ---- Extract all embeddings ----
    print("\n" + "=" * 60)
    print(f"Extracting embeddings from {len(labels)} images")
    print("=" * 60)

    embeddings = []
    timings = []
    errors = []

    for idx, row in tqdm(labels.iterrows(), total=len(labels), desc="Extracting"):
        try:
            t0 = time.time()
            serialized = image_to_tf_example(row["full_path"], args.image_size)
            emb = extract_image_embedding(vision_model, serialized)
            t1 = time.time()

            embeddings.append(emb.flatten())
            timings.append(t1 - t0)

        except Exception as e:
            errors.append({"file": row["image_id"], "error": str(e)})
            embeddings.append(None)  # Placeholder
            if len(errors) <= 3:
                print(f"\nError on {row['image_id']}: {e}")

    # Remove failed ones
    valid_mask = [e is not None for e in embeddings]
    embeddings = [e for e in embeddings if e is not None]
    metadata = labels[valid_mask].reset_index(drop=True)

    # ---- Save ----
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(args.output_dir / "embeddings.npy", embeddings_array)

    # Save metadata (drop full_path, it's machine-specific)
    metadata_out = metadata[["image_id", "file_path", "body_region", "has_fracture"]]
    metadata_out.to_csv(args.output_dir / "metadata.csv", index=False)

    stats = {
        "total_images": len(embeddings_array),
        "errors": len(errors),
        "embedding_shape": list(embeddings_array.shape),
        "mean_latency_ms": float(np.mean(timings) * 1000),
        "median_latency_ms": float(np.median(timings) * 1000),
        "p95_latency_ms": float(np.percentile(timings, 95) * 1000),
        "total_time_s": float(sum(timings)),
        "image_size": args.image_size,
    }
    with open(args.output_dir / "extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Embeddings: {args.output_dir / 'embeddings.npy'}  shape={embeddings_array.shape}")
    print(f"Metadata:   {args.output_dir / 'metadata.csv'}  ({len(metadata_out)} rows)")
    print(f"Errors:     {len(errors)}")
    print(f"Latency:    {stats['mean_latency_ms']:.0f} ms/image")
    print(f"Total time: {stats['total_time_s']:.0f} s ({stats['total_time_s']/60:.1f} min)")

    # ---- Text embeddings for zero-shot ----
    if not args.skip_text:
        print("\n" + "=" * 60)
        print("Extracting text embeddings for zero-shot")
        print("=" * 60)

        text_model = tf.saved_model.load(str(TEXT_MODEL_DIR))
        print("Text model loaded ✓")

        prompts = [
            "fracture", "bone fracture", "fracture present", "acute fracture",
            "fracture line", "cortical disruption", "broken bone",
            "normal", "no fracture", "normal bone", "intact cortex",
            "no acute fracture", "normal bone structure",
        ]

        text_embeddings = {}
        for prompt in tqdm(prompts, desc="Text embeddings"):
            try:
                text_emb = extract_text_embedding(text_model, prompt)
                text_embeddings[prompt] = text_emb
                if prompt == "fracture":
                    print(f"  '{prompt}' → shape={text_emb.shape}")
            except Exception as e:
                print(f"  Error on '{prompt}': {e}")

        np.savez(args.output_dir / "fracture_text_embeddings.npz", **text_embeddings)
        print(f"Saved {len(text_embeddings)} text embeddings")

    print("\n" + "=" * 60)
    print("DONE ✓")
    print("=" * 60)
    print(f"Next: python scripts/02_zero_shot.py")
    print(f"      python scripts/03_linear_probe.py")


if __name__ == "__main__":
    main()