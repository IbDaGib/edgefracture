#!/usr/bin/env python3
"""
Experiment 2: Linear Probe + Data Efficiency Curve

The KEY experiment for the Novel Task Prize:
  "How many labeled fracture X-rays does CXR Foundation need to become
   clinically useful for musculoskeletal screening?"

Method:
  - Extract frozen CXR Foundation embeddings (from Experiment 0)
  - Train logistic regression classifiers on increasing data subsets
  - 5-fold stratified cross-validation at each level
  - Plot AUC vs. training set size

Target result: AUC > 0.80 with ≤100 labeled examples

Usage:
    python scripts/03_linear_probe.py
"""

import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    precision_recall_curve, average_precision_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

EMBEDDINGS_DIR = Path("results/embeddings")
RESULTS_DIR = Path("results/linear_probe")

# Data efficiency: train on these subset sizes
SUBSET_SIZES = [10, 25, 50, 100, 250, 500, "all"]
N_FOLDS = 5
RANDOM_SEED = 42


# ============================================================
# Linear Probe Training
# ============================================================

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
) -> dict:
    """Train a logistic regression on frozen embeddings and evaluate."""
    
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",  # Handle class imbalance
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train_s, y_train)
    
    # Predict
    y_score = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)
    
    # Metrics
    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = 0.5
    
    try:
        ap = average_precision_score(y_test, y_score)
    except ValueError:
        ap = 0.0
    
    report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Fracture"],
        output_dict=True,
        zero_division=0,
    )
    
    return {
        "auc": auc,
        "average_precision": ap,
        "sensitivity": report["Fracture"]["recall"],
        "specificity": report["Normal"]["recall"],
        "precision": report["Fracture"]["precision"],
        "f1": report["Fracture"]["f1-score"],
        "y_score": y_score,
        "y_pred": y_pred,
        "model": clf,
        "scaler": scaler,
    }


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int | str,
    n_folds: int = N_FOLDS,
) -> dict:
    """
    Run stratified k-fold cross-validation with a limited training set.
    
    For subset experiments: randomly sample n_train examples from the
    training fold, evaluate on the full test fold.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Subsample training set if requested
        if isinstance(n_train, int) and n_train < len(X_train):
            rng = np.random.RandomState(RANDOM_SEED + fold)
            # Stratified subsample
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            
            # Maintain approximate class ratio
            n_pos = max(1, int(n_train * len(pos_idx) / len(y_train)))
            n_neg = n_train - n_pos
            n_pos = min(n_pos, len(pos_idx))
            n_neg = min(n_neg, len(neg_idx))
            
            sub_pos = rng.choice(pos_idx, n_pos, replace=False)
            sub_neg = rng.choice(neg_idx, n_neg, replace=False)
            sub_idx = np.concatenate([sub_pos, sub_neg])
            
            X_train = X_train[sub_idx]
            y_train = y_train[sub_idx]
        
        # Train and evaluate
        result = train_linear_probe(X_train, y_train, X_test, y_test)
        result["fold"] = fold
        result["n_train_actual"] = len(y_train)
        result["n_test"] = len(y_test)
        fold_results.append(result)
    
    # Aggregate across folds
    aucs = [r["auc"] for r in fold_results]
    return {
        "n_train": n_train if isinstance(n_train, int) else len(X),
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "min_auc": np.min(aucs),
        "max_auc": np.max(aucs),
        "mean_sensitivity": np.mean([r["sensitivity"] for r in fold_results]),
        "mean_specificity": np.mean([r["specificity"] for r in fold_results]),
        "mean_precision": np.mean([r["precision"] for r in fold_results]),
        "mean_f1": np.mean([r["f1"] for r in fold_results]),
        "fold_results": fold_results,
    }


def run_per_region(
    X: np.ndarray,
    y: np.ndarray,
    body_regions: np.ndarray,
) -> dict:
    """Run full cross-validation per body region."""
    region_results = {}
    
    for region in np.unique(body_regions):
        if region == "unknown":
            continue
        
        mask = body_regions == region
        X_r, y_r = X[mask], y[mask]
        
        if len(y_r) < 20 or y_r.sum() < 5:
            print(f"  Skipping {region}: too few samples ({len(y_r)} total, {y_r.sum()} fractures)")
            continue
        
        # Use fewer folds if data is scarce
        n_folds = min(N_FOLDS, int(y_r.sum()))
        n_folds = max(2, n_folds)
        
        result = run_cross_validation(X_r, y_r, "all", n_folds=n_folds)
        region_results[region] = result
        print(f"  {region:>10}: AUC={result['mean_auc']:.4f} ± {result['std_auc']:.4f} "
              f"(n={len(y_r)}, fractures={y_r.sum()})")
    
    return region_results


# ============================================================
# Plotting
# ============================================================

def plot_data_efficiency_curve(efficiency_results: list, output_dir: Path):
    """Plot the headline chart: AUC vs number of training examples."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes = [r["n_train"] for r in efficiency_results]
    means = [r["mean_auc"] for r in efficiency_results]
    stds = [r["std_auc"] for r in efficiency_results]
    
    # Plot with error bands
    ax.plot(sizes, means, "o-", color="#2c3e50", linewidth=2.5, markersize=8, zorder=3)
    ax.fill_between(
        sizes,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2, color="#3498db",
    )
    
    # Reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random chance (0.5)")
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Clinical target (0.8)")
    
    # Annotate points
    for x, y, s in zip(sizes, means, stds):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
    
    ax.set_xscale("log")
    ax.set_xlabel("Number of Labeled Training Examples", fontsize=12)
    ax.set_ylabel("AUC (5-Fold CV)", fontsize=12)
    ax.set_title("CXR Foundation → Fracture Detection: Data Efficiency Curve\n"
                 "How few labels does a chest X-ray model need to detect MSK fractures?",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    
    # Add x-axis tick labels
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    
    plt.tight_layout()
    plt.savefig(output_dir / "data_efficiency_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_dir / 'data_efficiency_curve.png'}")


def plot_per_region_results(region_results: dict, output_dir: Path):
    """Plot per-body-region AUC comparison."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    regions = sorted(region_results.keys())
    aucs = [region_results[r]["mean_auc"] for r in regions]
    stds = [region_results[r]["std_auc"] for r in regions]
    
    colors = ["#e74c3c" if a < 0.65 else "#f39c12" if a < 0.75 else "#2ecc71" for a in aucs]
    
    bars = ax.bar(regions, aucs, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5)
    
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Clinical target")
    
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{auc:.3f}", ha="center", fontsize=11, fontweight="bold")
    
    ax.set_ylabel("AUC (Cross-Validated)", fontsize=12)
    ax.set_title("Linear Probe Fracture Detection by Body Region\n"
                 "CXR Foundation (trained on chest X-rays only) → MSK Fractures",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_region_auc.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'per_region_auc.png'}")


def plot_combined_summary(efficiency_results: list, region_results: dict, output_dir: Path):
    """Create the combined summary figure for the writeup."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Panel A: Data efficiency curve
    ax = axes[0]
    sizes = [r["n_train"] for r in efficiency_results]
    means = [r["mean_auc"] for r in efficiency_results]
    stds = [r["std_auc"] for r in efficiency_results]
    
    ax.plot(sizes, means, "o-", color="#2c3e50", linewidth=2.5, markersize=8)
    ax.fill_between(sizes, [m-s for m,s in zip(means,stds)],
                    [m+s for m,s in zip(means,stds)], alpha=0.2, color="#3498db")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled Examples")
    ax.set_ylabel("AUC")
    ax.set_title("A) Data Efficiency", fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.grid(True, alpha=0.3)
    
    # Panel B: Per-region AUC
    ax = axes[1]
    regions = sorted(region_results.keys())
    aucs = [region_results[r]["mean_auc"] for r in regions]
    colors = ["#e74c3c" if a < 0.65 else "#f39c12" if a < 0.75 else "#2ecc71" for a in aucs]
    ax.bar(regions, aucs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.4)
    ax.set_ylabel("AUC")
    ax.set_title("B) Per-Region Performance", fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    for i, (r, a) in enumerate(zip(regions, aucs)):
        ax.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel C: Sensitivity vs Specificity at full data
    ax = axes[2]
    full_result = efficiency_results[-1]  # "all" data
    metrics = {
        "Sensitivity": full_result["mean_sensitivity"],
        "Specificity": full_result["mean_specificity"],
        "Precision": full_result["mean_precision"],
        "F1": full_result["mean_f1"],
        "AUC": full_result["mean_auc"],
    }
    bars = ax.bar(metrics.keys(), metrics.values(),
                  color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_title("C) Full-Data Performance", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("EdgeFracture: CXR Foundation Transfer to Musculoskeletal Fracture Detection",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "combined_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'combined_summary.png'}")


# ============================================================
# Save Best Model for Deployment
# ============================================================

def _calibrate_temperature(logits, y_true):
    """Find the temperature T that minimizes log-loss on held-out data.

    Temperature scaling: p = sigmoid(logit / T).
    A higher T softens probabilities toward 0.5 (less confident).
    """
    def neg_log_loss(T):
        probs = 1.0 / (1.0 + np.exp(-logits / T))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return log_loss(y_true, probs)

    result = minimize_scalar(neg_log_loss, bounds=(0.1, 20.0), method="bounded")
    return result.x


def save_deployment_model(X, y, output_dir: Path):
    """Train deployment model on 80% of data, evaluate on 20% held-out set.

    The held-out set is used to:
      1. Validate the actual deployed model (separate from CV metrics).
      2. Calibrate the temperature parameter for probability scaling.

    Saves model, scaler, calibrated temperature, and held-out metrics to pkl.
    """
    import pickle

    # --- Hold out 20% for final evaluation (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED,
    )
    print(f"\n  Deployment split: {len(y_train)} train / {len(y_test)} held-out test")
    print(f"    Train: {y_train.sum()} fractures, {(1-y_train).sum()} normal")
    print(f"    Test:  {y_test.sum()} fractures, {(1-y_test).sum()} normal")

    # --- Train on the training portion only ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    clf.fit(X_train_s, y_train)

    # --- Evaluate on held-out test set ---
    y_score = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)

    holdout_auc = roc_auc_score(y_test, y_score)
    holdout_ap = average_precision_score(y_test, y_score)
    holdout_report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Fracture"],
        output_dict=True,
        zero_division=0,
    )

    holdout_metrics = {
        "auc": holdout_auc,
        "average_precision": holdout_ap,
        "sensitivity": holdout_report["Fracture"]["recall"],
        "specificity": holdout_report["Normal"]["recall"],
        "precision": holdout_report["Fracture"]["precision"],
        "f1": holdout_report["Fracture"]["f1-score"],
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    # --- Calibrate temperature on held-out set ---
    logits = clf.decision_function(X_test_s)
    temperature = float(_calibrate_temperature(logits, y_test))

    # Compute calibrated log-loss for reporting
    calibrated_probs = 1.0 / (1.0 + np.exp(-logits / temperature))
    uncalibrated_ll = log_loss(y_test, np.clip(y_score, 1e-7, 1 - 1e-7))
    calibrated_ll = log_loss(y_test, np.clip(calibrated_probs, 1e-7, 1 - 1e-7))

    # --- Save everything ---
    model_path = output_dir / "fracture_probe.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "scaler": scaler,
            "temperature": temperature,
            "holdout_metrics": holdout_metrics,
        }, f)

    # --- Print results, clearly distinguishing CV vs held-out ---
    print(f"\n  Deployment model saved: {model_path}")
    print(f"  Model size: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"\n  NOTE: The cross-validation metrics above (e.g. AUC=0.882) were")
    print(f"  computed on separate CV folds of the full dataset. The metrics")
    print(f"  below are from the HELD-OUT test set for the actual deployed model.")
    print(f"\n  --- Held-out evaluation (deployment model) ---")
    print(f"    AUC:               {holdout_auc:.4f}")
    print(f"    Average Precision:  {holdout_ap:.4f}")
    print(f"    Sensitivity:        {holdout_metrics['sensitivity']:.4f}")
    print(f"    Specificity:        {holdout_metrics['specificity']:.4f}")
    print(f"    Precision:          {holdout_metrics['precision']:.4f}")
    print(f"    F1:                 {holdout_metrics['f1']:.4f}")
    print(f"\n  --- Temperature calibration ---")
    print(f"    Optimal temperature:  {temperature:.4f}")
    print(f"    Log-loss (uncalib.):   {uncalibrated_ll:.4f}")
    print(f"    Log-loss (calibrated): {calibrated_ll:.4f}")

    return clf, scaler


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    emb_path = args.embeddings_dir / "embeddings.npy"
    meta_path = args.embeddings_dir / "metadata.csv"
    
    if not emb_path.exists():
        print("Embeddings not found. Run scripts/01_extract_embeddings.py first.")
        return
    
    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
    
    # Filter to labeled images
    mask = metadata["has_fracture"].isin([0, 1])
    X = embeddings[mask.values]
    y = metadata.loc[mask, "has_fracture"].values.astype(int)
    body_regions = metadata.loc[mask, "body_region"].values
    
    print(f"Labeled: {len(y)} images ({y.sum()} fractures, {(1-y).sum()} normal)")
    print(f"Class balance: {y.mean():.1%} fracture, {1-y.mean():.1%} normal")
    print(f"Regions: {dict(zip(*np.unique(body_regions, return_counts=True)))}")
    
    # ========================================
    # Experiment 2a: Data Efficiency Curve
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 2a: DATA EFFICIENCY CURVE")
    print("="*60)
    
    efficiency_results = []
    
    for n_train in SUBSET_SIZES:
        if isinstance(n_train, int) and n_train >= len(X):
            continue  # Skip if subset larger than dataset
        
        label = str(n_train) if isinstance(n_train, int) else "all"
        print(f"\n--- Training with {label} examples ---")
        
        result = run_cross_validation(X, y, n_train)
        efficiency_results.append(result)
        
        print(f"  AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
        print(f"  Sensitivity: {result['mean_sensitivity']:.4f}")
        print(f"  Specificity: {result['mean_specificity']:.4f}")
    
    # ========================================
    # Experiment 2b: Per-Region Performance
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 2b: PER-REGION PERFORMANCE")
    print("="*60)
    
    region_results = run_per_region(X, y, body_regions)
    
    # ========================================
    # Generate Plots
    # ========================================
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_data_efficiency_curve(efficiency_results, args.output_dir)
    if region_results:
        plot_per_region_results(region_results, args.output_dir)
        plot_combined_summary(efficiency_results, region_results, args.output_dir)
    
    # ========================================
    # Save Results
    # ========================================
    
    # Clean results for JSON serialization
    json_results = {
        "data_efficiency": [
            {k: v for k, v in r.items() if k != "fold_results"}
            for r in efficiency_results
        ],
        "per_region": {
            region: {k: v for k, v in r.items() if k != "fold_results"}
            for region, r in region_results.items()
        },
    }
    
    with open(args.output_dir / "linear_probe_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # ========================================
    # Save Deployment Model
    # ========================================
    print("\n" + "="*60)
    print("SAVING DEPLOYMENT MODEL")
    print("="*60)
    
    clf, scaler = save_deployment_model(X, y, args.output_dir)
    
    # ========================================
    # Summary Table (for writeup)
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY TABLE (for writeup)")
    print("="*60)
    print(f"\n{'Training Examples':>20} {'AUC':>10} {'Sensitivity':>12} {'Specificity':>12}")
    print("-" * 58)
    for r in efficiency_results:
        n = r["n_train"]
        print(f"{n:>20} {r['mean_auc']:>10.4f} {r['mean_sensitivity']:>12.4f} {r['mean_specificity']:>12.4f}")
    
    if region_results:
        print(f"\n{'Body Region':>20} {'AUC':>10} {'N Images':>10} {'N Fractures':>12}")
        print("-" * 56)
        for region in sorted(region_results.keys()):
            r = region_results[region]
            fr = r["fold_results"][0] if r["fold_results"] else {}
            # Get from original data
            rmask = body_regions == region
            print(f"{region:>20} {r['mean_auc']:>10.4f} {rmask.sum():>10} {y[rmask].sum():>12}")
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("Next step: python scripts/04_edge_benchmarks.py")


if __name__ == "__main__":
    main()
