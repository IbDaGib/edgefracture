# Proposed Additions — Prioritized Review

> Assessed against the actual EdgeFracture codebase as of 2026-02-22.
> Items marked **ALREADY DONE** require no work. Remaining items are ranked by impact.

---

## ALREADY IMPLEMENTED (No Work Needed)

These proposals already exist in `app/app.py` and can be skipped entirely:

### ~~1. Programmatic safety disclaimers via regex~~
**Status: DONE** — `safety_gate()` (line ~360) scans MedGemma output with regex for overconfident language (`diagnose`, `confirm`, `definitely`, `certainly`, `"the fracture is"`), italicizes flagged terms, and appends a mandatory disclaimer to every response. No work needed.

### ~~2. Programmatic source/metadata appending~~
**Status: DONE** — `append_metadata_block()` (line ~390) programmatically appends a structured block with CXR Foundation confidence %, body region, triage level, model names/versions, and inference latency. Never relies on MedGemma to self-report numbers. No work needed.

### ~~10. Patient context field in Gradio~~
**Status: DONE** — Clinical context textbox exists in the Triage tab (512 char max). Input is sanitized by `_sanitize_clinical_context()` which strips 12 injection patterns, caps length, and removes prompt-breaking characters. Context is fed to MedGemma prompts alongside the fracture score. No work needed.

### ~~6. Structured JSON output from MedGemma~~
**Status: DONE** — `generate_report()` now attempts JSON-first prompts (`CLINICAL_JSON_PROMPT`, `PATIENT_JSON_PROMPT`), parses model output, and renders both narrative markdown and machine-readable JSON in the UI. Falls back to narrative output when JSON parsing fails.

### ~~8. MedGemma visual safety auditor~~
**Status: DONE** — `run_safety_audit()` performs a MedGemma vision pass after CXR classification, computes concordance (`CONCORDANT` / `DISCORDANT` / `UNCERTAIN`), and surfaces findings in a dedicated cross-check card.

---

## REMAINING ITEMS — Ranked Most to Least Important

### #1. Publish linear probe on HuggingFace
**Original rank: 4 | Effort: ~15 min | Impact: HIGH**

`results/linear_probe/fracture_probe.joblib` and `region_classifier.joblib` exist locally with SHA-256 integrity files. Upload both with a model card documenting:
- Training data: FracAtlas (4,083 MSK X-rays, 717 fractures)
- Architecture: LogisticRegression on 88,064-dim CXR Foundation embeddings
- Performance: 0.882 AUC (5-fold CV), per-region breakdowns with bootstrap CIs
- Temperature calibration: T=3.49

**Why #1:** Judges explicitly reward open-weight models. No competitor has a CXR Foundation transfer model published. This is a unique deliverable with near-zero risk.

**How it works here:** The `.joblib` files contain `{"model": LogisticRegression, "scaler": StandardScaler, "temperature": float}`. The SHA-256 sidecar files provide integrity verification. A HuggingFace model card can pull metrics directly from `results/linear_probe/linear_probe_results.json`.

---

### #2. Open writeup with a human story
**Original rank: 5 | Effort: ~5 min | Impact: HIGH**

One sentence framing: "Missed fractures are the #1 diagnostic error in emergency departments, affecting millions in regions with no radiologist."

**Why #2:** Zero technical risk, maximum narrative impact. The README currently opens with technical details. A human hook makes judges care before they evaluate.

**How it works here:** Add to `README.md` as the opening paragraph, before the current technical introduction.

---

### #3. Cite the 45-image TB result
**Original rank: 6 | Effort: ~5 min | Impact: HIGH**

Frame the data efficiency curve (`results/linear_probe/linear_probe_results.json`) as extending Google's published research: "Prior work demonstrated extreme data efficiency within CXR Foundation's training domain. EdgeFracture is the first evaluation of whether this transfers to musculoskeletal anatomy."

**Why #3:** The data efficiency experiment already shows 0.80 AUC with ~500 examples. Citing Google's own TB result positions this as a natural extension of their work, which judges will appreciate.

**How it works here:** The data is already in `linear_probe_results.json` with subset sizes [10, 25, 50, 100, 250, 500, 4024] and corresponding AUCs. Add the citation to the writeup/README alongside the existing efficiency curve.

---

### #4. Mirror Google's data efficiency chart format
**Original rank: 7 | Effort: ~20 min | Impact: MEDIUM-HIGH**

Reformat `data_efficiency_curve.png` to match Google's axes: AUC (y) vs. training data percentage (x). The current chart uses absolute training size on a log scale.

**Why #4:** Visual pattern-matching. Judges will instantly recognize their own evaluation format and see your result as a peer contribution rather than an independent experiment.

**How it works here:** Modify `scripts/03_linear_probe.py` visualization section. The data is already computed — this is purely a chart formatting change. Convert absolute counts [10, 25, 50, 100, 250, 500, 4024] to percentages of total (4024).

---

### #5. Conditional MedGemma invocation (auto-skip for Green)
**Original rank: 3 | Effort: ~15 min | Impact: MEDIUM**

Current behavior: `step_triage()` auto-generates both clinician and patient reports after triage for all score bands. This proposal is to skip report generation for GREEN triage (< 0.40 probability) to cut end-to-end latency.

**Why #5:** Auto-generation improves UX but adds avoidable latency for low-suspicion cases. Skipping MedGemma on green cases would preserve fast triage while reducing GPU contention and response time.

**How it works here:** In `app/app.py`, branch inside `step_triage()` before the `generate_report()` calls. For green cases, return a concise low-risk message and bypass MedGemma report generation.

---

### #7. DICOM-to-JPEG preprocessor
**Original rank: 9 | Effort: ~1-2 hours | Impact: MEDIUM**

`pydicom>=2.4.0` is already in `requirements.txt` but unused. Add a preprocessor that reads DICOM files, extracts pixel data, normalizes windowing, and converts to the grayscale PNG format CXR Foundation expects.

**Why #7:** Moves the project from "research demo" to "works with real medical equipment." The DICOM → PNG conversion is straightforward with pydicom. Frame as "Edge-Native DICOM Ingestion."

**How it works here:** Add a `_preprocess_dicom()` method to `FractureClassifier` in `app/app.py`. Modify the Gradio image upload to accept `.dcm` files. The pipeline after pixel extraction is identical — PIL Image → `_preprocess_image()` → embedding → classification.

```python
import pydicom
def _preprocess_dicom(self, dicom_path: str) -> Image.Image:
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array.astype(float)
    # Apply windowing if present
    # Normalize to 0-255
    # Return as PIL Image
```

---

### #9. 8-bit quantization instead of 4-bit
**Original rank: 12 | Effort: ~1-2 hours | Impact: LOW**

Currently using Q4_K_M quantization for MedGemma via Ollama (~2.5GB). Upgrading to 8-bit would improve text quality.

**Why #9 (lowest priority):** MedGemma now does vision tasks (region detection + visual safety audit), so higher precision could improve robustness. But the memory tradeoff is still severe on 8GB Jetson: CXR Foundation + MedGemma 8-bit + OS overhead leaves little headroom and may increase OOM risk.

**How it works here:** Change the Ollama model quantization. Test memory usage on Jetson: `CXR Foundation (~1GB) + MedGemma 8-bit (~4.5GB) + OS overhead (~1.5GB) = ~7GB`. Tight but possibly feasible. Would need actual Jetson testing to confirm.

---

## Summary Table

| Priority | Item | Effort | Status | Impact |
|----------|------|--------|--------|--------|
| -- | Safety disclaimers | -- | ALREADY DONE | -- |
| -- | Metadata appending | -- | ALREADY DONE | -- |
| -- | Patient context field | -- | ALREADY DONE | -- |
| -- | Structured JSON output | -- | ALREADY DONE | -- |
| -- | MedGemma safety auditor | -- | ALREADY DONE | -- |
| 1 | Publish probe on HuggingFace | 15 min | TODO | HIGH |
| 2 | Human story opening | 5 min | TODO | HIGH |
| 3 | Cite 45-image TB result | 5 min | TODO | HIGH |
| 4 | Mirror Google's chart format | 20 min | TODO | MEDIUM-HIGH |
| 5 | Conditional MedGemma (Green skip) | 15 min | TODO | MEDIUM |
| 7 | DICOM preprocessor | 1-2 hr | TODO | MEDIUM |
| 9 | 8-bit quantization | 1-2 hr | TODO | LOW |

**Total remaining work:** Items 1-5 are ~60 minutes and cover the highest-impact improvements. Items 7 and 9 add extra polish only after the core deliverables are done.

## What NOT to Do (Confirmed)

- Don't add RAG — MedGemma interprets a score, not guidelines
- Don't add audio/extra modalities — scope creep kills hackathons
- Don't fake "agentic" framing — the two-model pipeline is honest and clean
- Don't build multi-agent systems — complexity without value
- Don't add orchestration frameworks — unnecessary abstraction
