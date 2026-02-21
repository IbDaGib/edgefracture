#!/usr/bin/env python3
"""
Download and organize the FracAtlas dataset.

FracAtlas: 4,083 MSK X-ray images with 717 fracture cases
Body regions: Hand (1,538), Leg (2,272), Hip (338), Shoulder (349)
License: CC BY 4.0
Source: Abedeen et al., 2023 — Scientific Data (Nature)
"""

import os
import sys
import json
import shutil
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# FracAtlas is hosted on Figshare
# Update this URL if needed — check https://figshare.com/articles/dataset/FracAtlas/22717985
FRACATLAS_URL = "https://figshare.com/ndownloader/articles/22717985/versions/2"
DATA_DIR = Path("data/fracatlas")


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    resp = requests.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def organize_dataset(raw_dir: Path):
    """
    Organize FracAtlas into a clean structure.
    
    Expected FracAtlas structure after extraction:
    - images/ (all X-ray images)
    - Annotations/
      - COCO JSON annotations with fracture labels and body regions
    
    We'll create:
    - data/fracatlas/images/        (all images)
    - data/fracatlas/labels.csv     (image_id, body_region, has_fracture)
    """
    import pandas as pd
    
    # Look for the annotation file
    annotation_paths = list(raw_dir.rglob("*.json"))
    print(f"Found annotation files: {[str(p) for p in annotation_paths]}")
    
    # Try to find the COCO-format annotations
    coco_file = None
    for p in annotation_paths:
        try:
            with open(p) as f:
                data = json.load(f)
            if "images" in data and "annotations" in data:
                coco_file = p
                break
        except (json.JSONDecodeError, KeyError):
            continue
    
    if coco_file:
        print(f"Using COCO annotations from: {coco_file}")
        with open(coco_file) as f:
            coco = json.load(f)
        
        # Build lookup: image_id -> image info
        img_lookup = {img["id"]: img for img in coco["images"]}
        
        # Build lookup: image_id -> has annotations (fracture)
        fractured_ids = set()
        for ann in coco.get("annotations", []):
            fractured_ids.add(ann["image_id"])
        
        # Create labels
        records = []
        for img in coco["images"]:
            # Try to determine body region from filename or path
            fname = img.get("file_name", "")
            body_region = classify_body_region(fname)
            records.append({
                "image_id": img["id"],
                "file_name": fname,
                "body_region": body_region,
                "has_fracture": 1 if img["id"] in fractured_ids else 0,
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            })
        
        df = pd.DataFrame(records)
        labels_path = DATA_DIR / "labels.csv"
        df.to_csv(labels_path, index=False)
        print(f"Saved labels to {labels_path}")
        print(f"Total images: {len(df)}")
        print(f"Fracture cases: {df['has_fracture'].sum()}")
        print(f"Body region distribution:\n{df['body_region'].value_counts()}")
    else:
        print("WARNING: Could not find COCO annotations. You may need to manually organize.")
        print("Check the extracted files and update this script accordingly.")
        # Create a fallback that scans image directories
        create_labels_from_directory(raw_dir)


def classify_body_region(filename: str) -> str:
    """Classify body region from FracAtlas filename conventions."""
    fname_lower = filename.lower()
    if "hand" in fname_lower or "finger" in fname_lower or "wrist" in fname_lower:
        return "hand"
    elif "leg" in fname_lower or "ankle" in fname_lower or "knee" in fname_lower or "tibia" in fname_lower:
        return "leg"
    elif "hip" in fname_lower or "pelvis" in fname_lower:
        return "hip"
    elif "shoulder" in fname_lower or "humerus" in fname_lower:
        return "shoulder"
    else:
        return "unknown"


def create_labels_from_directory(raw_dir: Path):
    """Fallback: create labels by scanning directory structure."""
    import pandas as pd
    
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    records = []
    
    for img_path in raw_dir.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            # Infer fracture status from directory name
            parent_parts = [p.lower() for p in img_path.parts]
            has_fracture = 1 if any("fracture" in p for p in parent_parts) else 0
            body_region = classify_body_region(str(img_path))
            
            records.append({
                "file_name": str(img_path.relative_to(raw_dir)),
                "body_region": body_region,
                "has_fracture": has_fracture,
            })
    
    if records:
        df = pd.DataFrame(records)
        labels_path = DATA_DIR / "labels.csv"
        df.to_csv(labels_path, index=False)
        print(f"Created labels from directory scan: {len(df)} images")
        print(f"Fracture cases: {df['has_fracture'].sum()}")
    else:
        print("ERROR: No images found. Check the dataset structure.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "fracatlas_raw.zip"
    
    # Step 1: Download
    if not zip_path.exists():
        print("Downloading FracAtlas dataset...")
        print(f"URL: {FRACATLAS_URL}")
        print("(If this fails, download manually from https://figshare.com/articles/dataset/FracAtlas/22717985)")
        try:
            download_file(FRACATLAS_URL, zip_path)
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nManual download instructions:")
            print("1. Go to https://figshare.com/articles/dataset/FracAtlas/22717985")
            print("2. Download the dataset ZIP")
            print(f"3. Place it at: {zip_path}")
            print("4. Re-run this script")
            sys.exit(1)
    else:
        print(f"Found existing download: {zip_path}")
    
    # Step 2: Extract
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        print(f"Extracted to {raw_dir}")
    
    # Step 3: Organize
    print("Organizing dataset...")
    organize_dataset(raw_dir)
    
    print("\nDone! Dataset ready at:", DATA_DIR)
    print("\nNext step: python scripts/01_extract_embeddings.py")


if __name__ == "__main__":
    main()
