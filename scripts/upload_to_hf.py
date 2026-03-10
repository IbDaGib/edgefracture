#!/usr/bin/env python3
"""Upload fracture probe to HuggingFace Hub."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "ibdagib/edgefracture-cxr-fracture-probe"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES_TO_UPLOAD = {
    "results/linear_probe/fracture_probe.joblib": "fracture_probe.joblib",
    "results/linear_probe/fracture_probe.joblib.sha256": "fracture_probe.joblib.sha256",
    "model_card/README.md": "README.md",
}


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Not authenticated. Run `huggingface-cli login` or set HF_TOKEN.")
        sys.exit(1)

    api = HfApi(token=token)

    # Verify all files exist before uploading
    for local_path in FILES_TO_UPLOAD:
        full_path = PROJECT_ROOT / local_path
        if not full_path.exists():
            print(f"Error: {full_path} not found.")
            sys.exit(1)

    # Create repo (no-op if it already exists)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    # Upload each file
    for local_path, repo_path in FILES_TO_UPLOAD.items():
        full_path = PROJECT_ROOT / local_path
        api.upload_file(
            path_or_fileobj=str(full_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"Uploaded {local_path} -> {repo_path}")

    print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
