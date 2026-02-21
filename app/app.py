#!/usr/bin/env python3
"""
EdgeFracture — Gradio Demo Application

Portable musculoskeletal fracture triage running on Jetson Orin Nano.
Two HAI-DEF models: CXR Foundation (screening) + MedGemma 4B (report generation).

Usage:
    python app/app.py [--port 7860] [--share]
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import gradio as gr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Configuration
# ============================================================

PROBE_PATH = Path("results/linear_probe/fracture_probe.pkl")
MODEL_DIR = Path("models/cxr-foundation")
RESULTS_DIR = Path("results")

# Triage thresholds
TRIAGE_THRESHOLDS = {
    "high": 0.7,    # Red — high suspicion
    "moderate": 0.4, # Yellow — uncertain
    # Below 0.4 = Green — likely normal
}

TRIAGE_COLORS = {
    "HIGH": "#e74c3c",
    "MODERATE": "#f39c12",
    "LOW": "#2ecc71",
}

SAFETY_DISCLAIMER = (
    "⚠️ **AI-Assisted Screening Tool** — This is NOT a diagnosis. "
    "All findings must be confirmed by a qualified clinician or radiologist. "
    "This tool is a research prototype and is not FDA-cleared."
)

# ============================================================
# Model Loading
# ============================================================

class EdgeFractureEngine:
    """Main inference engine combining CXR Foundation + MedGemma."""
    
    def __init__(self):
        self.cxr_model = None
        self.linear_probe = None
        self.scaler = None
        self.medgemma_available = False
        self.device = "cpu"
        
        self._load_models()
    
    def _load_models(self):
        """Load all models."""
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Device: {self.device}")
        
        # Load CXR Foundation
        try:
            self._load_cxr_foundation()
        except Exception as e:
            print(f"WARNING: CXR Foundation not loaded: {e}")
            print("Will use linear probe only (requires pre-extracted embeddings)")
        
        # Load linear probe
        try:
            self._load_linear_probe()
        except Exception as e:
            print(f"WARNING: Linear probe not loaded: {e}")
        
        # Check MedGemma
        try:
            import ollama
            models = ollama.list()
            for m in models.get("models", []):
                if "medgemma" in m.get("name", "").lower():
                    self.medgemma_model = m["name"]
                    self.medgemma_available = True
                    print(f"MedGemma available: {self.medgemma_model}")
                    break
            if not self.medgemma_available:
                print("WARNING: MedGemma not found in Ollama")
        except ImportError:
            print("WARNING: ollama package not installed")
    
    def _load_cxr_foundation(self):
        """Load CXR Foundation model."""
        import torch
        from torchvision import transforms
        
        # Store transform for later use
        self.transform = transforms.Compose([
            transforms.Resize((1280, 1280)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        # Try to load the model
        try:
            from transformers import AutoModel
            self.cxr_model = AutoModel.from_pretrained(
                str(MODEL_DIR), trust_remote_code=True
            ).to(self.device).eval()
            print("CXR Foundation loaded successfully")
        except Exception as e:
            print(f"CXR Foundation loading failed: {e}")
            self.cxr_model = None
    
    def _load_linear_probe(self):
        """Load pre-trained linear probe."""
        if PROBE_PATH.exists():
            with open(PROBE_PATH, "rb") as f:
                data = pickle.load(f)
            self.linear_probe = data["model"]
            self.scaler = data["scaler"]
            print(f"Linear probe loaded from {PROBE_PATH}")
        else:
            print(f"Linear probe not found at {PROBE_PATH}")
            print("Run scripts/03_linear_probe.py first")
    
    def extract_embedding(self, image: Image.Image) -> np.ndarray | None:
        """Extract CXR Foundation embedding from an image."""
        import torch
        
        if self.cxr_model is None:
            return None
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.cxr_model(img_tensor)
            
            # Handle different output formats
            if hasattr(outputs, "image_embeds"):
                emb = outputs.image_embeds
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1)
            elif isinstance(outputs, dict):
                emb = list(outputs.values())[0]
            else:
                emb = outputs
        
        return emb.cpu().numpy().flatten()
    
    def classify_fracture(self, embedding: np.ndarray) -> dict:
        """Run linear probe classification on embedding."""
        if self.linear_probe is None or self.scaler is None:
            return {"error": "Linear probe not loaded"}
        
        X = self.scaler.transform(embedding.reshape(1, -1))
        prob = self.linear_probe.predict_proba(X)[0, 1]
        
        # Determine triage level
        if prob >= TRIAGE_THRESHOLDS["high"]:
            triage = "HIGH"
        elif prob >= TRIAGE_THRESHOLDS["moderate"]:
            triage = "MODERATE"
        else:
            triage = "LOW"
        
        return {
            "fracture_probability": float(prob),
            "triage_level": triage,
            "triage_color": TRIAGE_COLORS[triage],
        }
    
    def generate_report(self, classification: dict, body_region: str = "unknown") -> str:
        """Generate clinical report via MedGemma."""
        if not self.medgemma_available:
            return self._fallback_report(classification, body_region)
        
        prob = classification["fracture_probability"]
        triage = classification["triage_level"]
        
        prompt = f"""You are a musculoskeletal radiology assistant. An AI screening system 
has analyzed a {body_region} X-ray and produced the following results:

- Fracture probability: {prob*100:.0f}% ({triage})
- Classification: {'fracture_detected' if prob > 0.5 else 'no_fracture_detected'}
- Confidence: {'high' if abs(prob - 0.5) > 0.3 else 'moderate' if abs(prob - 0.5) > 0.15 else 'low'}

Based on these screening results, provide:

1. CLINICAL SUMMARY (2-3 sentences): Describe what this screening result suggests, 
   including possible fracture types common in this body region and recommended next steps.

2. URGENCY LEVEL: Classify as one of:
   - URGENT: Likely displaced or unstable fracture — splint and refer
   - MODERATE: Possible non-displaced fracture — further imaging recommended  
   - LOW: Low suspicion — clinical correlation advised

3. PATIENT EXPLANATION (2-3 sentences): Explain the result in simple, 
   reassuring language suitable for the patient.

Important: This is an AI screening result, not a diagnosis. Always 
recommend professional radiologist review."""

        try:
            import ollama
            response = ollama.generate(
                model=self.medgemma_model,
                prompt=prompt,
                options={"num_predict": 400, "temperature": 0.3},
            )
            return response.get("response", "Error generating report")
        except Exception as e:
            return f"Report generation error: {e}\n\n{self._fallback_report(classification, body_region)}"
    
    def _fallback_report(self, classification: dict, body_region: str) -> str:
        """Template-based fallback when MedGemma is unavailable."""
        prob = classification["fracture_probability"]
        triage = classification["triage_level"]
        
        if triage == "HIGH":
            return f"""**CLINICAL SUMMARY:** AI screening detected a high probability ({prob*100:.0f}%) of fracture 
in the {body_region} X-ray. Common fracture types in this region should be considered. 
Recommend urgent clinical review and possible further imaging (CT).

**URGENCY:** URGENT — Splint and refer for specialist evaluation.

**PATIENT EXPLANATION:** The screening tool has flagged a possible fracture in your 
{body_region} X-ray that needs further evaluation. Please don't be alarmed — a doctor 
will review this finding and determine the next steps for your care."""
        
        elif triage == "MODERATE":
            return f"""**CLINICAL SUMMARY:** AI screening shows moderate probability ({prob*100:.0f}%) of fracture 
in the {body_region} X-ray. Subtle or non-displaced fractures cannot be excluded. 
Recommend further evaluation with additional views or CT if clinically indicated.

**URGENCY:** MODERATE — Further imaging recommended.

**PATIENT EXPLANATION:** The screening shows some findings that need a closer look. 
This doesn't necessarily mean there's a fracture, but your doctor will want to 
review this carefully and may order additional imaging to be thorough."""
        
        else:
            return f"""**CLINICAL SUMMARY:** AI screening shows low probability ({prob*100:.0f}%) of fracture 
in the {body_region} X-ray. No obvious fracture identified by the screening system. 
Clinical correlation is advised.

**URGENCY:** LOW — Clinical correlation advised.

**PATIENT EXPLANATION:** The screening didn't find strong signs of a fracture in your 
{body_region} X-ray. Your doctor will confirm this result. If you're still having 
pain, they may want to do a follow-up check."""


# ============================================================
# Gradio Interface
# ============================================================

def create_triage_card(classification: dict) -> str:
    """Create HTML triage card."""
    prob = classification["fracture_probability"]
    triage = classification["triage_level"]
    color = classification["triage_color"]
    
    icon = "🔴" if triage == "HIGH" else "🟡" if triage == "MODERATE" else "🟢"
    
    return f"""
    <div style="background: {color}20; border: 3px solid {color}; border-radius: 12px; 
                padding: 20px; text-align: center; margin: 10px 0;">
        <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 36px; font-weight: bold; color: {color};">{prob*100:.1f}%</div>
        <div style="font-size: 14px; color: #666; margin-top: 4px;">Fracture Probability</div>
        <div style="font-size: 24px; font-weight: bold; color: {color}; margin-top: 8px;">
            {triage} SUSPICION
        </div>
    </div>
    """


def create_app(engine: EdgeFractureEngine):
    """Create the Gradio application."""
    
    def process_xray(image, body_region, generate_report_flag):
        """Main processing function."""
        if image is None:
            return "Please upload an X-ray image.", "", ""
        
        start_time = time.time()
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("L")
        else:
            pil_image = image.convert("L")
        
        # Step 1: Extract embedding
        embedding = engine.extract_embedding(pil_image)
        
        if embedding is None:
            return (
                "<div style='color: red; padding: 20px;'>CXR Foundation model not loaded. "
                "Please check model installation.</div>",
                "",
                f"Embedding extraction failed. Time: {time.time()-start_time:.2f}s"
            )
        
        embed_time = time.time() - start_time
        
        # Step 2: Classify
        classification = engine.classify_fracture(embedding)
        
        if "error" in classification:
            return (
                f"<div style='color: red; padding: 20px;'>{classification['error']}</div>",
                "",
                f"Classification failed. Time: {time.time()-start_time:.2f}s"
            )
        
        classify_time = time.time() - start_time
        
        # Step 3: Triage card (instant)
        triage_html = create_triage_card(classification)
        
        # Step 4: Generate report (optional, slower)
        report = ""
        if generate_report_flag:
            report = engine.generate_report(classification, body_region)
            report_time = time.time() - start_time
        else:
            report_time = classify_time
        
        # Timing info
        timing = (
            f"⏱️ **Timing:** Embedding: {embed_time*1000:.0f}ms | "
            f"Classification: {(classify_time-embed_time)*1000:.0f}ms | "
            f"Total: {report_time:.1f}s"
        )
        
        return triage_html, report, timing
    
    def process_batch(files, body_region):
        """Process multiple X-rays and return triage summary."""
        if not files:
            return "No files uploaded."
        
        results = []
        for f in files:
            img = Image.open(f.name).convert("L")
            embedding = engine.extract_embedding(img)
            
            if embedding is not None:
                classification = engine.classify_fracture(embedding)
                results.append({
                    "file": Path(f.name).name,
                    "probability": classification["fracture_probability"],
                    "triage": classification["triage_level"],
                })
            else:
                results.append({
                    "file": Path(f.name).name,
                    "probability": None,
                    "triage": "ERROR",
                })
        
        # Sort by probability (highest first)
        results.sort(key=lambda x: x["probability"] or 0, reverse=True)
        
        # Format as table
        header = "| # | File | Probability | Triage |\n|---|------|------------|--------|\n"
        rows = ""
        for i, r in enumerate(results, 1):
            icon = "🔴" if r["triage"] == "HIGH" else "🟡" if r["triage"] == "MODERATE" else "🟢"
            prob = f"{r['probability']*100:.1f}%" if r["probability"] else "N/A"
            rows += f"| {i} | {r['file']} | {prob} | {icon} {r['triage']} |\n"
        
        return header + rows
    
    # Build the interface
    with gr.Blocks(
        title="EdgeFracture — Fracture Triage",
        theme=gr.themes.Soft(),
        css="""
        .disclaimer { 
            background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; 
            padding: 12px; margin: 10px 0; font-size: 13px; 
        }
        """
    ) as app:
        gr.Markdown(
            """
            # 🦴 EdgeFracture — Portable Fracture Triage
            **AI-powered musculoskeletal fracture screening on edge hardware**
            
            Upload an X-ray image to get an instant triage assessment. 
            Uses CXR Foundation (chest X-ray model) repurposed for fracture detection 
            + MedGemma for clinical report generation.
            """
        )
        gr.HTML(f'<div class="disclaimer">{SAFETY_DISCLAIMER}</div>')
        
        with gr.Tabs():
            # Tab 1: Single Image
            with gr.TabItem("🩻 Single Image Triage"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload X-ray",
                            type="pil",
                            height=400,
                        )
                        body_region = gr.Dropdown(
                            choices=["hand", "leg", "hip", "shoulder", "unknown"],
                            value="unknown",
                            label="Body Region (if known)",
                        )
                        generate_report = gr.Checkbox(
                            label="Generate Clinical Report (slower — 30-90s)",
                            value=False,
                        )
                        submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        triage_output = gr.HTML(label="Triage Result")
                        report_output = gr.Markdown(label="Clinical Report")
                        timing_output = gr.Markdown(label="Performance")
                
                submit_btn.click(
                    fn=process_xray,
                    inputs=[image_input, body_region, generate_report],
                    outputs=[triage_output, report_output, timing_output],
                )
            
            # Tab 2: Batch Mode
            with gr.TabItem("📁 Batch Triage"):
                gr.Markdown("Upload multiple X-rays to get a prioritized triage summary.")
                batch_input = gr.File(
                    label="Upload X-ray Files",
                    file_count="multiple",
                    file_types=["image"],
                )
                batch_region = gr.Dropdown(
                    choices=["hand", "leg", "hip", "shoulder", "mixed"],
                    value="mixed",
                    label="Body Region",
                )
                batch_btn = gr.Button("🔍 Analyze Batch", variant="primary")
                batch_output = gr.Markdown(label="Triage Summary")
                
                batch_btn.click(
                    fn=process_batch,
                    inputs=[batch_input, batch_region],
                    outputs=[batch_output],
                )
            
            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
                    ## About EdgeFracture
                    
                    **What:** A portable fracture screening tool that repurposes a chest X-ray 
                    foundation model to detect musculoskeletal fractures across body regions.
                    
                    **Models:**
                    - **CXR Foundation** (`google/cxr-foundation`) — Trained on 821K chest X-rays. 
                      We demonstrate it can transfer to fracture detection in hands, legs, hips, 
                      and shoulders — anatomy it has never seen.
                    - **MedGemma 4B** — Generates clinical triage reports and patient explanations.
                    
                    **Hardware:** NVIDIA Jetson Orin Nano 8GB ($249) — fully offline, no cloud needed.
                    
                    **Dataset:** FracAtlas (4,083 MSK X-rays, CC BY 4.0)
                    
                    **Competition:** MedGemma Impact Challenge 2026 (Kaggle / Google Health AI)
                    
                    ---
                    
                    **Disclaimer:** This is a research prototype. Not for clinical use. 
                    Not FDA-cleared. All outputs require professional medical review.
                    """
                )
        
        # Footer with device info
        gr.Markdown(
            """
            ---
            <center>
            EdgeFracture • Running on NVIDIA Jetson Orin Nano 8GB • 
            CXR Foundation + MedGemma 4B • Fully Offline
            </center>
            """,
            elem_classes=["footer"],
        )
    
    return app


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    print("="*60)
    print("EdgeFracture — Loading models...")
    print("="*60)
    
    engine = EdgeFractureEngine()
    app = create_app(engine)
    
    print("\n" + "="*60)
    print(f"Launching on port {args.port}...")
    print("="*60)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
