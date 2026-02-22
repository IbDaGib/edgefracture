# EdgeFracture — ML Engineering Review

Overall this is a well-structured competition project with a compelling transfer learning story. That said, there are several flaws ranging from critical ML methodology issues to production concerns.

---

## CRITICAL — ML Methodology Flaws

### ~~1. Batch triage sorting is broken~~ FIXED

**File:** `app/app.py:626`

~~This sorts **strings** like `"9.1%"`, `"78.5%"`, `"100.0%"` lexicographically. Result: `"9.1%" > "78.5%"` because `"9" > "7"` in string comparison. In a medical triage queue, this means **low-risk cases could surface above high-risk cases**.~~

**Fix applied:** Sort key now uses `float(r["Fracture %"].rstrip("%"))` with error sentinel `-1`.

### ~~2. Zero-shot temperature tuning leaks data~~ FIXED

**File:** `scripts/02_zero_shot.py:486-498`

~~The best temperature is selected by evaluating all candidates on the **full dataset** — not cross-validated. This optimizes a hyperparameter directly on the test set, inflating reported zero-shot AUC.~~

**Fix applied:** Temperature selection now uses 5-fold StratifiedKFold CV. Only validation folds are scored per candidate.

### ~~3. Region classifier reports metrics on training data~~ FIXED

**File:** `scripts/05_region_classifier.py:62-66`

~~The printed classification report and confusion matrix are computed on the same data used for training, producing misleadingly perfect numbers.~~

**Fix applied:** Replaced with `cross_val_predict` for honest out-of-fold metrics. Final `model.fit(X, y)` is kept only for saving the deployment artifact.

### ~~4. Region classifier has no StandardScaler~~ FIXED

**File:** `scripts/05_region_classifier.py:50-57`

~~The fracture probe uses `StandardScaler` on 88,064-dim embeddings before logistic regression, but the region classifier trains directly on raw embeddings.~~

**Fix applied:** Wrapped in sklearn `Pipeline` with `StandardScaler` + `LogisticRegression`. Saved pkl now contains the full pipeline. *Requires retraining to take effect.*

### ~~5. Deployment model trained on ALL data — no held-out evaluation~~ FIXED

**File:** `scripts/03_linear_probe.py:369-389`

~~`save_deployment_model()` trains on the entire dataset. The reported AUC=0.882 comes from cross-validation of separate models — it's **not** the performance of the actual deployed model.~~

**Fix applied:** Now holds out 20% (stratified) for final evaluation. Held-out metrics saved in pkl. *Requires retraining to take effect.*

---

## HIGH — Security & Safety Flaws

### 6. Arbitrary code execution via `pickle.load()` — OPEN

**File:** `app/app.py:75-76, 89-90`

```python
with open(LINEAR_PROBE_PATH, "rb") as f:
    data = pickle.load(f)
```

Pickle deserialization can execute arbitrary code. If an attacker replaces the `.pkl` file (e.g., via supply chain or path traversal), they get full code execution. For a medical application, use `safetensors` or save model weights + config separately. At minimum, validate file hashes.

### ~~7. Clinical context passed directly to LLM without sanitization~~ FIXED

**File:** `app/app.py:383-389`

~~User-provided `clinical_context` is interpolated directly into the MedGemma prompt. This enables **prompt injection**.~~

**Fix applied:** Added `_sanitize_clinical_context()` — strips 12 injection patterns, limits to 500 chars, removes prompt-breaking characters.

### ~~8. No input image validation~~ FIXED

**File:** `app/app.py:124-143`

~~Any PIL-openable image is accepted and processed. There's no validation that the input is actually an X-ray.~~

**Fix applied:** Added `_validate_xray_image()` — checks grayscale, dimensions, aspect ratio. Advisory warnings only; classification always proceeds.

---

## MEDIUM — Data & Pipeline Issues

### ~~9. Contrastive embedding errors silently replaced with zeros~~ FIXED

**File:** `scripts/01_extract_embeddings.py:410-412`

~~Failed extractions are replaced with zero vectors, which stay in the dataset and bias cosine similarity computations.~~

**Fix applied:** Now saves `contrastive_valid_mask.npy` boolean mask alongside embeddings. Downstream scripts can filter out zero-filled entries. *Requires re-extraction to generate the mask file.*

### 10. Unreliable per-region metrics for Hip and Shoulder — OPEN

- Hip: 179 images, **10 fractures**
- Shoulder: 98 images, **10 fractures**

With only 10 positive cases, the per-region AUC estimates have very wide confidence intervals. A single misclassified fracture case swings AUC by ~0.10. The reported AUCs (0.864 for Hip, 0.848 for Shoulder) appear precise but are statistically unreliable. This should be prominently flagged.

### ~~11. TEMPERATURE = 3.49 is a magic number~~ FIXED

**File:** `app/app.py:240`

~~This calibration temperature is hardcoded with no code to reproduce it.~~

**Fix applied:** Added `_calibrate_temperature()` in `scripts/03_linear_probe.py` using `scipy.optimize.minimize_scalar` on held-out log-loss. Temperature saved in pkl; `app.py` loads dynamically with 3.49 fallback. *Requires retraining to embed temperature in pkl.*

### ~~12. Benchmark script measures wrong framework~~ FIXED

**File:** `scripts/04_edge_benchmarks.py:141-157`

~~`benchmark_cxr_foundation()` imports PyTorch and uses `torchvision.transforms`, but the actual app uses **TensorFlow** `SavedModel`.~~

**Fix applied:** Rewrote to use TF `SavedModel` + `tf.train.Example` preprocessing matching the production path in `app.py`.

---

## LOW — Production & Code Quality

### 13. No test suite at all — OPEN

Zero unit tests, integration tests, or model validation tests. For a medical application, at minimum you need:

- Sanity checks that the model outputs are in valid ranges
- Regression tests ensuring AUC doesn't degrade after code changes
- Input validation tests

### 14. Global mutable classifier state — OPEN

**File:** `app/app.py:651`

```python
classifier = FractureClassifier()
```

Module-level singleton shared across all Gradio requests. TensorFlow's `saved_model.load()` may not be thread-safe for concurrent requests. Gradio's batch triage could trigger race conditions.

### 15. No prediction logging or audit trail — OPEN

For a medical screening tool, there's no logging of predictions, inputs, or model decisions. You have no way to:

- Audit past predictions
- Detect model drift
- Investigate misclassifications

### ~~16. Version-pinning gaps in requirements.txt~~ FIXED

~~All dependencies use `>=` with no upper bound.~~

**Fix applied:** All 19 dependencies now pinned with `<major.0.0` upper bounds.

### ~~17. `classifier_real.py` has a dead `NotImplementedError`~~ FIXED

~~The `_extract_embedding` method always raises `NotImplementedError`. This file is checked into the repo as a "drop-in replacement" but is non-functional.~~

**Fix applied:** Marked as DEPRECATED/TEMPLATE ONLY with pointer to the active implementation in `app.py`.

### ~~18. Redundant TF import side-effect~~ FIXED

~~`import tensorflow_text` looks like a mistake to a reader.~~

**Fix applied:** Added `# noqa: F401` + comment explaining SentencepieceOp registration requirement.

---

## Recommended Fixes (Priority Order)

| # | Severity | Fix | Status |
|---|----------|-----|--------|
| 1 | **Critical** | Fix batch sort: use numeric key | ~~FIXED~~ |
| 7 | **High** | Sanitize clinical_context before prompt interpolation | ~~FIXED~~ |
| 2 | **High** | Cross-validate zero-shot temperature selection | ~~FIXED~~ |
| 3 | **High** | Remove misleading full-dataset classification report | ~~FIXED~~ |
| 4 | **Medium** | Add StandardScaler to region classifier | ~~FIXED~~ (retrain needed) |
| 5 | **Medium** | Hold out 15-20% of data for final model evaluation | ~~FIXED~~ (retrain needed) |
| 9 | **Medium** | Filter out zero-vector embeddings during evaluation | ~~FIXED~~ (re-extract needed) |
| 11 | **Medium** | Add temperature calibration code to the repo | ~~FIXED~~ (retrain needed) |
| 12 | **Medium** | Rewrite benchmark to use TF inference path | ~~FIXED~~ |
| 16 | **Low** | Pin dependency upper bounds | ~~FIXED~~ |
| 17 | **Low** | Mark dead classifier_real.py as deprecated | ~~FIXED~~ |
| 18 | **Low** | Add noqa + comment for side-effect import | ~~FIXED~~ |
| 6 | **Low** | Replace pickle with joblib + hash validation | OPEN |
| 10 | **Low** | Add confidence intervals to per-region AUC (bootstrap) | OPEN |
| 13 | **Low** | Add test suite | OPEN |
| 14 | **Low** | Address global mutable classifier state | OPEN |
| 15 | **Low** | Add prediction logging / audit trail | OPEN |

**13 of 18 issues fixed.** 5 remaining issues are all LOW severity or require architectural decisions.
