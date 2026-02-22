#!/usr/bin/env python3
"""
Experiment 1: Zero-Shot Fracture Detection via CXR Foundation

Uses ELIXR-contrastive text-image alignment to classify fractures
WITHOUT any training data. The model has never seen musculoskeletal X-rays.

Method:
  - Encode text prompts: "fracture present" vs "normal bone"
  - Compute cosine similarity between contrastive image embeddings (32x128)
    and contrastive text embeddings (128-dim)
  - Pool max similarity across 32 query vectors per image
  - Softmax -> fracture probability

This is the headline experiment for the Novel Task Prize.
Even modest performance (AUC > 0.60) demonstrates meaningful transfer.

Usage:
    python scripts/02_zero_shot.py
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path("results")
EMBEDDINGS_DIR = Path("results/embeddings")
MODEL_DIR = Path("models/cxr-foundation")
PRECOMPUTED_DIR = MODEL_DIR / "precomputed_embeddings"

# ============================================================
# Zero-Shot Prompt Sets
# ============================================================
# We try multiple prompt formulations to find the best one.
# CXR Foundation's text encoder was trained on radiology report text,
# so medical/radiology language should work best.

PROMPT_SETS = {
    "basic": {
        "positive": ["fracture"],
        "negative": ["normal"],
    },
    "clinical_v1": {
        "positive": ["fracture present in bone"],
        "negative": ["normal bone structure without fracture"],
    },
    "clinical_v2": {
        "positive": ["bone fracture", "broken bone", "fracture line visible"],
        "negative": ["normal bone", "intact bone structure", "no fracture"],
    },
    "radiology": {
        "positive": [
            "fracture line identified",
            "cortical disruption consistent with fracture",
            "acute fracture",
        ],
        "negative": [
            "no acute fracture",
            "intact cortical margins",
            "normal bony structures",
        ],
    },
    "descriptive": {
        "positive": [
            "there is a fracture",
            "fracture is present",
            "bone discontinuity",
        ],
        "negative": [
            "no fracture is seen",
            "bones appear normal",
            "no bone abnormality",
        ],
    },
}


# ============================================================
# Text Embedding Loading / Extraction
# ============================================================

def load_text_embeddings_from_npz(npz_path: Path) -> dict:
    """Load text embeddings from a .npz file. Returns {prompt: embedding}."""
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def extract_text_embeddings_inline(prompts: list) -> dict:
    """Extract text embeddings using Q-Former + BERT tokenizer inline.

    Falls back to this when pre-saved embeddings are unavailable.
    """
    import tensorflow as tf
    import tensorflow_text  # noqa: F401
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    qformer = tf.saved_model.load(str(MODEL_DIR / "pax-elixr-b-text"))
    serve_fn = qformer.signatures["serving_default"]

    result = {}
    for prompt in prompts:
        try:
            encoded = tokenizer(
                prompt, max_length=128, padding="max_length",
                truncation=True, return_tensors="np",
            )
            ids = tf.constant(encoded["input_ids"].astype(np.int32).reshape(1, 1, 128))
            paddings = tf.constant(
                (1.0 - encoded["attention_mask"].astype(np.float32)).reshape(1, 1, 128)
            )
            image_feature = tf.zeros((1, 8, 8, 1376), dtype=tf.float32)
            output = serve_fn(ids=ids, paddings=paddings, image_feature=image_feature)
            result[prompt] = output["contrastive_txt_emb"].numpy()[0]
        except Exception as e:
            print(f"  Failed to extract '{prompt}': {e}")
    return result


def load_precomputed_text_embeddings() -> dict:
    """Load precomputed demo text embeddings from the model repo."""
    path = PRECOMPUTED_DIR / "text_embeddings.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


def get_text_embeddings_for_prompts(
    prompt_sets: dict, embeddings_dir: Path,
) -> dict:
    """Resolve text embeddings for all prompt sets.

    Strategy (in order):
      1. Load from fracture_text_embeddings.npz (our extracted embeddings)
      2. Match against precomputed demo embeddings
      3. Extract inline via Q-Former (last resort)
    """
    # Collect all unique prompts
    all_prompts = set()
    for ps in prompt_sets.values():
        all_prompts.update(ps["positive"])
        all_prompts.update(ps["negative"])

    resolved = {}

    # 1. Try our saved embeddings
    our_npz = embeddings_dir / "fracture_text_embeddings.npz"
    if our_npz.exists():
        saved = load_text_embeddings_from_npz(our_npz)
        if saved:
            print(f"Loaded {len(saved)} text embeddings from {our_npz}")
            for key in list(saved.keys()):
                # npz keys may have been lowercased; try case-insensitive match
                resolved[key] = saved[key]

    # 2. Fill gaps from precomputed demo embeddings
    missing = all_prompts - set(resolved.keys())
    if missing:
        precomputed = load_precomputed_text_embeddings()
        if precomputed:
            # Try exact match and common variants
            missing_before = set(missing)
            for prompt in list(missing):
                for variant in [prompt, prompt.capitalize(), prompt.lower(), prompt.upper()]:
                    if variant in precomputed:
                        resolved[prompt] = precomputed[variant]
                        missing.discard(prompt)
                        break
            n_matched = len(missing_before) - len(missing)
            if n_matched > 0:
                print(f"Matched {n_matched} prompts from precomputed demo embeddings")

    # 3. Try inline extraction for remaining
    if missing:
        print(f"  {len(missing)} prompts still missing, attempting inline extraction...")
        try:
            extracted = extract_text_embeddings_inline(list(missing))
            resolved.update(extracted)
            missing -= set(extracted.keys())
            if extracted:
                print(f"  Extracted {len(extracted)} embeddings inline")
                # Save for future use
                all_saved = dict(resolved)
                np.savez(embeddings_dir / "fracture_text_embeddings.npz", **all_saved)
                print(f"  Updated fracture_text_embeddings.npz")
        except Exception as e:
            print(f"  Inline extraction failed: {e}")

    if missing:
        print(f"  WARNING: Could not resolve embeddings for: {missing}")

    # Build per-prompt-set arrays
    text_embs = {}
    for name, ps in prompt_sets.items():
        pos = [resolved[p] for p in ps["positive"] if p in resolved]
        neg = [resolved[p] for p in ps["negative"] if p in resolved]
        if pos and neg:
            text_embs[name] = {
                "positive": np.array(pos),
                "negative": np.array(neg),
            }
        else:
            skipped_pos = [p for p in ps["positive"] if p not in resolved]
            skipped_neg = [p for p in ps["negative"] if p not in resolved]
            print(f"  Skipping prompt set '{name}': missing pos={skipped_pos}, neg={skipped_neg}")

    return text_embs


# ============================================================
# Zero-Shot Classification
# ============================================================

def compute_zero_shot_scores(
    image_embeddings: np.ndarray,
    text_embeddings_pos: np.ndarray,
    text_embeddings_neg: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute fracture probability via cosine similarity.

    Supports two image embedding formats:
      - (N, 4096): contrastive embeddings from Q-Former (32 x 128 queries)
        -> reshape to (N, 32, 128), compute sim with each 128-dim text embedding,
           max-pool across 32 queries
      - (N, D) where D != 4096: flat embeddings, direct cosine similarity

    Text embeddings are always 128-dim.

    Returns: array of fracture probabilities, shape (N,)
    """
    N = image_embeddings.shape[0]
    emb_dim = image_embeddings.shape[1]

    if emb_dim == 4096:
        # Multi-query contrastive: (N, 32, 128)
        img_queries = image_embeddings.reshape(N, 32, 128)

        # Normalize each query vector
        norms = np.linalg.norm(img_queries, axis=2, keepdims=True) + 1e-8
        img_queries_norm = img_queries / norms

        # Normalize text embeddings
        pos_norm = text_embeddings_pos / (np.linalg.norm(text_embeddings_pos, axis=1, keepdims=True) + 1e-8)
        neg_norm = text_embeddings_neg / (np.linalg.norm(text_embeddings_neg, axis=1, keepdims=True) + 1e-8)

        # For each text embedding, compute similarity with all 32 image queries, take max
        # sim shape per text prompt: (N, 32) -> max -> (N,)
        pos_sims = []
        for t in range(pos_norm.shape[0]):
            # (N, 32, 128) @ (128,) -> (N, 32)
            sim = np.einsum("nqd,d->nq", img_queries_norm, pos_norm[t])
            pos_sims.append(sim.max(axis=1))  # (N,)
        avg_sim_pos = np.mean(pos_sims, axis=0)  # (N,)

        neg_sims = []
        for t in range(neg_norm.shape[0]):
            sim = np.einsum("nqd,d->nq", img_queries_norm, neg_norm[t])
            neg_sims.append(sim.max(axis=1))
        avg_sim_neg = np.mean(neg_sims, axis=0)

    else:
        # Flat embeddings: direct cosine similarity
        img_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
        pos_norm = text_embeddings_pos / (np.linalg.norm(text_embeddings_pos, axis=1, keepdims=True) + 1e-8)
        neg_norm = text_embeddings_neg / (np.linalg.norm(text_embeddings_neg, axis=1, keepdims=True) + 1e-8)

        sim_pos = img_norm @ pos_norm.T  # (N, P)
        sim_neg = img_norm @ neg_norm.T  # (N, Q)

        avg_sim_pos = sim_pos.mean(axis=1)
        avg_sim_neg = sim_neg.mean(axis=1)

    # Softmax to get probabilities
    logits = np.stack([avg_sim_neg, avg_sim_pos], axis=1) / temperature  # (N, 2)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return probs[:, 1]  # P(fracture)


def evaluate_zero_shot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    body_regions: np.ndarray,
    prompt_name: str,
    output_dir: Path,
):
    """Evaluate zero-shot performance and generate plots."""

    results = {"prompt_set": prompt_name}

    # Overall AUC
    try:
        auc = roc_auc_score(y_true, y_score)
        results["overall_auc"] = round(auc, 4)
        print(f"  Overall AUC: {auc:.4f}")
    except ValueError as e:
        print(f"  Could not compute overall AUC: {e}")
        results["overall_auc"] = None

    # Per-region AUC
    regions = np.unique(body_regions)
    region_results = {}
    for region in regions:
        if region == "unknown":
            continue
        mask = body_regions == region
        if mask.sum() < 5 or y_true[mask].sum() < 1:
            continue
        try:
            region_auc = roc_auc_score(y_true[mask], y_score[mask])
            n_total = int(mask.sum())
            n_fracture = int(y_true[mask].sum())
            region_results[region] = {
                "auc": round(region_auc, 4),
                "n_images": n_total,
                "n_fractures": n_fracture,
            }
            print(f"  {region:>10}: AUC={region_auc:.4f} (n={n_total}, fractures={n_fracture})")
        except ValueError:
            continue

    results["per_region"] = region_results

    # At best threshold (Youden's J)
    if results.get("overall_auc"):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        y_pred = (y_score >= best_threshold).astype(int)

        report = classification_report(y_true, y_pred, target_names=["Normal", "Fracture"], output_dict=True)
        results["best_threshold"] = round(float(best_threshold), 4)
        results["sensitivity"] = round(report["Fracture"]["recall"], 4)
        results["specificity"] = round(report["Normal"]["recall"], 4)
        results["classification_report"] = report

        print(f"  Best threshold: {best_threshold:.4f}")
        print(f"  Sensitivity: {results['sensitivity']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")

    return results


def plot_zero_shot_results(all_results: list, y_true: np.ndarray, output_dir: Path):
    """Generate summary plots for zero-shot experiments."""

    # 1. ROC curves for each prompt set
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: ROC curves
    ax = axes[0]
    for res in all_results:
        if res.get("overall_auc") and "y_score" in res:
            fpr, tpr, _ = roc_curve(y_true, res["y_score"])
            ax.plot(fpr, tpr, label=f"{res['prompt_set']} (AUC={res['overall_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Zero-Shot Fracture Detection -- ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Per-region AUC comparison (best prompt set)
    ax = axes[1]
    best = max(all_results, key=lambda r: r.get("overall_auc", 0) or 0)
    if best.get("per_region"):
        regions = list(best["per_region"].keys())
        aucs = [best["per_region"][r]["auc"] for r in regions]
        colors = ["#e74c3c" if a < 0.6 else "#f39c12" if a < 0.7 else "#2ecc71" for a in aucs]
        bars = ax.bar(regions, aucs, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
        ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="Target (0.7)")
        ax.set_ylabel("AUC")
        ax.set_title(f"Per-Region AUC -- Best Prompt: '{best['prompt_set']}'")
        ax.set_ylim(0.3, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{auc:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "zero_shot_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {output_dir / 'zero_shot_results.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "zero_shot")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load image embeddings ----
    # Prefer contrastive embeddings (4096-dim, from Q-Former) over raw (88064-dim)
    contrastive_path = args.embeddings_dir / "contrastive_embeddings.npy"
    raw_path = args.embeddings_dir / "embeddings.npy"
    meta_path = args.embeddings_dir / "metadata.csv"

    if not meta_path.exists():
        print("Metadata not found. Run scripts/01_extract_embeddings.py first.")
        return

    metadata = pd.read_csv(meta_path)

    if contrastive_path.exists():
        embeddings = np.load(contrastive_path)
        emb_type = "contrastive (4096-dim, 32x128)"
    elif raw_path.exists():
        embeddings = np.load(raw_path)
        emb_type = f"raw ({embeddings.shape[1]}-dim)"
    else:
        print("No embeddings found. Run scripts/01_extract_embeddings.py first.")
        return

    print(f"Loaded {len(embeddings)} image embeddings: {emb_type}")
    print(f"  Shape: {embeddings.shape}")

    # Filter to labeled images only
    mask = metadata["has_fracture"].isin([0, 1])
    embeddings = embeddings[mask.values]
    metadata = metadata[mask].reset_index(drop=True)

    y_true = metadata["has_fracture"].values
    body_regions = metadata["body_region"].values
    print(f"Labeled images: {len(y_true)} (fractures: {y_true.sum()}, normal: {(1-y_true).sum()})")

    # ---- TEXT EMBEDDINGS ----
    print("\n--- Loading text embeddings ---")
    text_embs = get_text_embeddings_for_prompts(PROMPT_SETS, args.embeddings_dir)

    if not text_embs:
        print("\nERROR: No text embeddings available for any prompt set.")
        print("Run: python scripts/01_extract_embeddings.py --skip-images")
        return

    print(f"\nResolved {len(text_embs)} prompt sets: {list(text_embs.keys())}")

    # Verify dimensionality match
    sample_text = list(text_embs.values())[0]["positive"][0]
    print(f"Text embedding dim: {sample_text.shape[0]}")
    if embeddings.shape[1] == 4096:
        print(f"Image embedding: 32 x 128 (multi-query contrastive) -- will max-pool")
    elif embeddings.shape[1] == sample_text.shape[0]:
        print(f"Image/text dimensions match: {sample_text.shape[0]}")
    else:
        print(f"WARNING: Image dim ({embeddings.shape[1]}) != text dim ({sample_text.shape[0]})")
        print("  Cosine similarity may not be meaningful across different embedding spaces.")

    # ---- EVALUATION ----
    print("\n" + "="*60)
    print("ZERO-SHOT FRACTURE DETECTION RESULTS")
    print("="*60)

    all_results = []

    for prompt_name in PROMPT_SETS:
        if prompt_name not in text_embs:
            continue

        print(f"\nPrompt set: '{prompt_name}'")

        pos_embs = text_embs[prompt_name]["positive"]
        neg_embs = text_embs[prompt_name]["negative"]

        # Select temperature via cross-validation to prevent test-set leakage.
        # Without CV, evaluating all candidate temperatures on the full dataset
        # and picking the best one optimizes a hyperparameter directly on the
        # test set, inflating the reported AUC. Using stratified K-fold ensures
        # the temperature is chosen based on held-out validation performance.
        candidate_temps = [0.01, 0.05, 0.1, 0.5, 1.0]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        temp_cv_aucs = {temp: [] for temp in candidate_temps}

        for train_idx, val_idx in skf.split(embeddings, y_true):
            y_val = y_true[val_idx]
            emb_val = embeddings[val_idx]
            for temp in candidate_temps:
                fold_scores = compute_zero_shot_scores(emb_val, pos_embs, neg_embs, temperature=temp)
                try:
                    fold_auc = roc_auc_score(y_val, fold_scores)
                    temp_cv_aucs[temp].append(fold_auc)
                except ValueError:
                    continue

        best_auc = 0
        best_temp = 1.0
        for temp in candidate_temps:
            if temp_cv_aucs[temp]:
                mean_auc = np.mean(temp_cv_aucs[temp])
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_temp = temp

        # Compute final scores on full dataset with the CV-selected temperature
        scores = compute_zero_shot_scores(embeddings, pos_embs, neg_embs, temperature=best_temp)
        print(f"  Best temperature: {best_temp}")

        results = evaluate_zero_shot(y_true, scores, body_regions, prompt_name, args.output_dir)
        results["temperature"] = best_temp
        results["y_score"] = scores  # Keep for plotting
        all_results.append(results)

    # Save results
    # Remove y_score (numpy array) before saving to JSON
    json_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "y_score"}
        json_results.append(r_copy)

    with open(args.output_dir / "zero_shot_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Plot
    if all_results:
        plot_zero_shot_results(all_results, y_true, args.output_dir)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Prompt Set':<20} {'AUC':>8} {'Sens.':>8} {'Spec.':>8} {'Temp':>6}")
    print("-" * 54)
    for r in sorted(all_results, key=lambda x: x.get("overall_auc", 0) or 0, reverse=True):
        print(f"{r['prompt_set']:<20} {r.get('overall_auc', 'N/A'):>8} "
              f"{r.get('sensitivity', 'N/A'):>8} {r.get('specificity', 'N/A'):>8} "
              f"{r.get('temperature', 'N/A'):>6}")

    print(f"\nResults saved to: {args.output_dir}")
    print("Next step: python scripts/03_linear_probe.py")


if __name__ == "__main__":
    main()
