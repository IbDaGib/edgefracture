# EdgeFracture — Execution Checklist

## ⚡ Priority Order (Do EXACTLY This)

The single biggest risk is CXR Foundation not loading properly. 
**Test model loading FIRST before building anything else.**

---

## TODAY (Day 1) — Models + Experiments

### Block 1: Environment (1 hour)
- [ ] SSH into Jetson / set up dev environment
- [ ] Run `bash setup_jetson.sh`
- [ ] Verify PyTorch sees CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Verify Ollama is running: `ollama list`

### Block 2: CXR Foundation — THE CRITICAL TEST (2 hours)
- [ ] Accept license at https://huggingface.co/google/cxr-foundation
- [ ] Run `python scripts/download_cxr_foundation.py`
- [ ] **IMMEDIATELY test loading + inference** — open a Python shell:
  ```python
  # Quick smoke test — adapt based on actual model files
  from pathlib import Path
  import torch
  from PIL import Image
  from torchvision import transforms
  
  # Check what files were downloaded
  for f in Path("models/cxr-foundation").rglob("*"):
      if f.is_file():
          print(f"{f.relative_to('models/cxr-foundation')} — {f.stat().st_size/1e6:.1f} MB")
  
  # Try loading
  from transformers import AutoModel
  model = AutoModel.from_pretrained("models/cxr-foundation", trust_remote_code=True)
  print("SUCCESS — model loaded")
  print(f"Type: {type(model)}")
  print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
  ```
- [ ] If loading fails → check HuggingFace model card for exact loading instructions
- [ ] If ARM64/Jetson fails → **FALLBACK: extract embeddings on Kaggle T4 notebook, copy .npy to Jetson**

### Block 3: FracAtlas Data (30 min)
- [ ] Run `python scripts/download_fracatlas.py`
- [ ] Verify: images exist, labels.csv created, body regions mapped
- [ ] Quick check: `wc -l data/fracatlas/labels.csv` (should be ~4,084 lines)

### Block 4: Extract Embeddings (1-2 hours)
- [ ] Run `python scripts/01_extract_embeddings.py`
- [ ] Verify: `results/embeddings/embeddings.npy` exists
- [ ] Check shape: should be (N, embedding_dim) — likely (4083, 4096) or similar
- [ ] Note the extraction latency per image

### Block 5: Experiments (2 hours)
- [ ] Run `python scripts/02_zero_shot.py` — get zero-shot AUC numbers
- [ ] Run `python scripts/03_linear_probe.py` — get data efficiency curve
- [ ] **Check the data_efficiency_curve.png** — this is your headline chart
- [ ] Note: Even if zero-shot AUC is ~0.55, the linear probe should be much better

### Block 6: MedGemma Setup (1 hour)
- [ ] Pull MedGemma: `ollama pull hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M`
- [ ] Test: `ollama run medgemma-4b-it "What is a Colles fracture?"`
- [ ] If 4B is unstable → try `medgemma-1.5-4b-it` Q4 variant
- [ ] Verify it fits in memory alongside CXR Foundation

---

## TOMORROW (Day 2) — App + Video

### Block 7: Gradio App (2 hours)
- [ ] Run `python app/app.py`
- [ ] Test single image flow: upload → triage card → report
- [ ] Test with FracAtlas images from each body region
- [ ] Fix any model loading or inference issues

### Block 8: Edge Benchmarks (1 hour)
- [ ] Run `python scripts/04_edge_benchmarks.py`
- [ ] Capture: CXR latency, MedGemma latency, memory, power
- [ ] Save benchmark_table.md for writeup

### Block 9: Record Demo Video (1.5 hours)
- [ ] Show the physical Jetson hardware
- [ ] Show boot → app launch → single image triage (instant result!)
- [ ] Show MedGemma report generation
- [ ] Show batch mode
- [ ] Show benchmark metrics
- [ ] Show the data efficiency curve chart
- [ ] Keep under 3 minutes
- [ ] SCRIPT IT before recording

### Block 10: Writeup (2 hours)
- [ ] Use Kaggle writeup template
- [ ] Page 1: Problem + solution overview + user journey
- [ ] Page 2: Technical approach + results (data efficiency curve + per-region AUC)
- [ ] Page 3: Edge benchmarks + impact + future work
- [ ] Include: combined_summary.png, benchmark table, architecture diagram

---

## CRITICAL FALLBACKS

| If This Fails... | Do This Instead |
|---|---|
| CXR Foundation won't load on Jetson | Extract embeddings on Kaggle T4 → copy .npy files → run everything else on Jetson |
| MedGemma crashes on Jetson | Use template-based reports (already built into app.py) |
| Zero-shot AUC is near random | Frame it: "zero-shot fails, but just 50 labels gets 0.80" — makes the data efficiency story STRONGER |
| FracAtlas annotations are weird | Manually inspect files, write a simple label parser |
| Jetson runs out of memory | Load one model at a time (CXR first, unload, then MedGemma) |

---

## FILES TO SUBMIT
1. ✅ GitHub repo (this code)
2. ✅ 3-min video (MP4)
3. ✅ 3-page writeup (Kaggle)
4. 🎁 Live demo URL (bonus — try ngrok on Jetson)
