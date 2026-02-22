# EdgeFracture — ML Engineering Review

Overall this is a well-structured competition project with a compelling transfer learning story. ~~That said, there are several flaws ranging from critical ML methodology issues to production concerns.~~ **All 18 identified issues have been fixed.**

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

**Fix applied:** Wrapped in sklearn `Pipeline` with `StandardScaler` + `LogisticRegression`. Saved model now contains the full pipeline.

### ~~5. Deployment model trained on ALL data — no held-out evaluation~~ FIXED

**File:** `scripts/03_linear_probe.py:369-389`

~~`save_deployment_model()` trains on the entire dataset. The reported AUC=0.882 comes from cross-validation of separate models — it's **not** the performance of the actual deployed model.~~

**Fix applied:** Now holds out 20% (stratified) for final evaluation. Held-out metrics saved in model file.

---

## HIGH — Security & Safety Flaws

### ~~6. Arbitrary code execution via `pickle.load()`~~ FIXED

**File:** `app/app.py:75-76, 89-90`

~~Pickle deserialization can execute arbitrary code. If an attacker replaces the `.pkl` file (e.g., via supply chain or path traversal), they get full code execution.~~

**Fix applied:** Switched to `joblib` serialization with SHA-256 hash sidecar verification. Training scripts write `.joblib` + `.sha256` files; app verifies hash before loading. Legacy `.pkl` fallback retained for backwards compatibility.

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

**Fix applied:** Now saves `contrastive_valid_mask.npy` boolean mask alongside embeddings. Downstream scripts can filter out zero-filled entries.

### ~~10. Unreliable per-region metrics for Hip and Shoulder~~ FIXED

~~With only 10 positive cases each, the per-region AUC estimates have very wide confidence intervals. The reported AUCs appear precise but are statistically unreliable.~~

**Fix applied:** Added `bootstrap_auc_ci()` (2000 resamples, 95% CI) to `scripts/03_linear_probe.py`. Regions with CI width > 0.15 are flagged as statistically unreliable. CIs are shown in the app's Evidence panel with an explanatory note for Hip and Shoulder.

### ~~11. TEMPERATURE = 3.49 is a magic number~~ FIXED

**File:** `app/app.py:240`

~~This calibration temperature is hardcoded with no code to reproduce it.~~

**Fix applied:** Added `_calibrate_temperature()` in `scripts/03_linear_probe.py` using `scipy.optimize.minimize_scalar` on held-out log-loss. Temperature saved in model file; `app.py` loads dynamically with 3.49 fallback.

### ~~12. Benchmark script measures wrong framework~~ FIXED

**File:** `scripts/04_edge_benchmarks.py:141-157`

~~`benchmark_cxr_foundation()` imports PyTorch and uses `torchvision.transforms`, but the actual app uses **TensorFlow** `SavedModel`.~~

**Fix applied:** Rewrote to use TF `SavedModel` + `tf.train.Example` preprocessing matching the production path in `app.py`.

---

## LOW — Production & Code Quality

### ~~13. No test suite at all~~ FIXED

~~Zero unit tests, integration tests, or model validation tests.~~

**Fix applied:** Added `tests/test_app.py` with 60 tests across 9 test classes: batch sort, sanitization, image validation, triage thresholds, confidence labels, safety gate, placeholder scores, context building. All pure Python — no TF or model files required.

### ~~14. Global mutable classifier state~~ FIXED

**File:** `app/app.py:651`

~~Module-level singleton shared across all Gradio requests. TensorFlow's `saved_model.load()` may not be thread-safe for concurrent requests.~~

**Fix applied:** Replaced bare `classifier = FractureClassifier()` with `get_classifier()` using double-checked locking via `threading.Lock`. Lazy initialization ensures thread-safe model loading.

### ~~15. No prediction logging or audit trail~~ FIXED

~~No logging of predictions, inputs, or model decisions.~~

**Fix applied:** Added `PredictionLogger` writing JSON-lines to `logs/predictions.jsonl` (10MB rotation, 5 backups). Records timestamp, probability, triage level, region, model, latency. Does not log images or raw clinical context (privacy). `logs/` added to `.gitignore`.

### ~~16. Version-pinning gaps in requirements.txt~~ FIXED

~~All dependencies use `>=` with no upper bound.~~

**Fix applied:** All dependencies now pinned with `<major.0.0` upper bounds.

### ~~17. `classifier_real.py` has a dead `NotImplementedError`~~ FIXED

~~The `_extract_embedding` method always raises `NotImplementedError`. This file is checked into the repo as a "drop-in replacement" but is non-functional.~~

**Fix applied:** Marked as DEPRECATED/TEMPLATE ONLY with pointer to the active implementation in `app.py`.

### ~~18. Redundant TF import side-effect~~ FIXED

~~`import tensorflow_text` looks like a mistake to a reader.~~

**Fix applied:** Added `# noqa: F401` + comment explaining SentencepieceOp registration requirement.

---

## Summary

| # | Severity | Fix | Status |
|---|----------|-----|--------|
| 1 | **Critical** | Fix batch sort: use numeric key | ~~FIXED~~ |
| 2 | **Critical** | Cross-validate zero-shot temperature selection | ~~FIXED~~ |
| 3 | **Critical** | Remove misleading full-dataset classification report | ~~FIXED~~ |
| 4 | **Critical** | Add StandardScaler to region classifier | ~~FIXED~~ |
| 5 | **Critical** | Hold out 15-20% of data for final model evaluation | ~~FIXED~~ |
| 6 | **High** | Replace pickle with joblib + hash validation | ~~FIXED~~ |
| 7 | **High** | Sanitize clinical_context before prompt interpolation | ~~FIXED~~ |
| 8 | **High** | Add input image validation | ~~FIXED~~ |
| 9 | **Medium** | Filter out zero-vector embeddings during evaluation | ~~FIXED~~ |
| 10 | **Medium** | Add confidence intervals to per-region AUC (bootstrap) | ~~FIXED~~ |
| 11 | **Medium** | Add temperature calibration code to the repo | ~~FIXED~~ |
| 12 | **Medium** | Rewrite benchmark to use TF inference path | ~~FIXED~~ |
| 13 | **Low** | Add test suite | ~~FIXED~~ |
| 14 | **Low** | Address global mutable classifier state | ~~FIXED~~ |
| 15 | **Low** | Add prediction logging / audit trail | ~~FIXED~~ |
| 16 | **Low** | Pin dependency upper bounds | ~~FIXED~~ |
| 17 | **Low** | Mark dead classifier_real.py as deprecated | ~~FIXED~~ |
| 18 | **Low** | Add noqa + comment for side-effect import | ~~FIXED~~ |
| 19 | **Critical** | Rename `file_name` to `file_path`, add `image_id` | ~~FIXED~~ |
| 20 | **High** | Add `_fmt()` helper for safe numeric formatting | ~~FIXED~~ |
| 21 | **High** | Add `tensorflow-text` to requirements.txt | ~~FIXED~~ |
| 22 | **Medium** | Filter invalid embeddings in zero-shot script | ~~FIXED~~ |
| 23 | **Medium** | Wrap torch import in try/except, add TF fallback | ~~FIXED~~ |
| 24 | **Low** | Fix `.pkl` → `.joblib` typo in print | ~~FIXED~~ |

**24 of 24 issues fixed.**

---

## NEW ISSUES (Round 2)

### ~~19. Column mismatch breaks pipeline from scratch~~ FIXED

**File:** `scripts/download_fracatlas.py:92,142-146`

~~`organize_dataset()` writes `file_name` but `01_extract_embeddings.py` expects `file_path` and `image_id`. Running the pipeline from a fresh download fails immediately.~~

**Fix applied:** Renamed `file_name` to `file_path` in both `organize_dataset()` and `create_labels_from_directory()`. Added missing `image_id` field to fallback path.

### ~~20. `format_benchmark_table` crashes on missing metrics~~ FIXED

**File:** `scripts/04_edge_benchmarks.py:422-438`

~~F-string format specs like `:.0f` crash with `TypeError` when the dict `.get()` returns the default `'N/A'` string.~~

**Fix applied:** Added `_fmt()` helper that safely formats numeric values or returns `'N/A'`. All format specs in the table replaced with `_fmt()` calls.

### ~~21. Missing `tensorflow-text` in requirements.txt~~ FIXED

**File:** `requirements.txt`

~~Fresh installs fail because `tensorflow-text` is imported by the CXR Foundation pipeline but not listed as a dependency.~~

**Fix applied:** Added `tensorflow-text>=2.16.0,<3.0.0` after the `tensorflow` line.

### ~~22. Zero-shot doesn't filter invalid embeddings~~ FIXED

**File:** `scripts/02_zero_shot.py:~435`

~~Zero-vector embeddings (from failed extractions, see issue #9) are included in cosine similarity computation, biasing results.~~

**Fix applied:** After loading embeddings, loads `contrastive_valid_mask.npy` and filters out invalid entries. Prints count of filtered entries; warns if mask file is missing.

### ~~23. `profile_memory()` imports torch unconditionally~~ FIXED

**File:** `scripts/04_edge_benchmarks.py:379-400`

~~`import torch` at the top of `profile_memory()` crashes on TensorFlow-only environments (e.g., Jetson with only TF installed).~~

**Fix applied:** Wrapped `import torch` in `try/except ImportError`. Added TensorFlow GPU detection as fallback. System RAM stats via psutil are always returned.

### ~~24. `.pkl` typo in region classifier print~~ FIXED

**File:** `scripts/05_region_classifier.py:111`

~~Print statement says `.pkl` but the model is saved as `.joblib`.~~

**Fix applied:** Changed `.pkl` to `.joblib` in the print string.

---

## Summary (Round 2)

| # | Severity | Fix | Status |
|---|----------|-----|--------|
| 19 | **Critical** | Rename `file_name` to `file_path`, add `image_id` | ~~FIXED~~ |
| 20 | **High** | Add `_fmt()` helper for safe numeric formatting | ~~FIXED~~ |
| 21 | **High** | Add `tensorflow-text` to requirements.txt | ~~FIXED~~ |
| 22 | **Medium** | Filter invalid embeddings in zero-shot script | ~~FIXED~~ |
| 23 | **Medium** | Wrap torch import in try/except, add TF fallback | ~~FIXED~~ |
| 24 | **Low** | Fix `.pkl` → `.joblib` typo in print | ~~FIXED~~ |
