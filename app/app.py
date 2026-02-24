#!/usr/bin/env python3
"""
EdgeFracture — Portable Musculoskeletal Fracture Triage
MedGemma Impact Challenge 2026

Stepped workflow: Upload → Triage → Report → Evidence
Runs fully offline on Jetson Orin Nano ($249)

Two HAI-DEF models:
  - CXR Foundation (google/cxr-foundation) — image embeddings + fracture classification
  - MedGemma 1.5 4B (google/medgemma-1.5-4b-it) — clinical report generation + body region detection + visual safety audit
"""

import argparse
import io
import json
import threading
from pathlib import Path

import gradio as gr
import requests
from PIL import Image

try:
    from .classifier import FractureClassifier
    from .config import (
        CXR_MODEL_PATH,
        CXR_SOFT_LOCK_MARGIN,
        LINEAR_PROBE_PATH,
        MEDGEMMA_MODEL,
        OLLAMA_BASE_URL,
        SAFETY_AUDIT_ENABLED,
        SAFETY_AUDIT_TIMEOUT,
        TRIAGE_THRESHOLDS,
        VALIDATED_REGIONS,
    )
    from .evidence import (
        get_model_transparency_info as _get_model_transparency_info,
        get_performance_context as _get_performance_context,
    )
    from .prediction_logger import prediction_logger
    from .reporting import (
        CLINICAL_JSON_PROMPT as _CLINICAL_JSON_PROMPT,
        CLINICAL_PROMPT as _CLINICAL_PROMPT,
        PATIENT_JSON_PROMPT as _PATIENT_JSON_PROMPT,
        PATIENT_PROMPT as _PATIENT_PROMPT,
        SAFETY_DISCLAIMER as _SAFETY_DISCLAIMER,
        STRUCTURED_JSON_FIELDS_CLINICAL as _STRUCTURED_JSON_FIELDS_CLINICAL,
        STRUCTURED_JSON_FIELDS_PATIENT as _STRUCTURED_JSON_FIELDS_PATIENT,
        append_metadata_block as _append_metadata_block_impl,
        build_audit_findings_section as _build_audit_findings_section_impl,
        build_clinical_summary_section as _build_clinical_summary_section_impl,
        build_context_section as _build_context_section_impl,
        clean_medgemma_output as _clean_medgemma_output_impl,
        generate_report as _generate_report_impl,
        normalize_clinical_structured_json as _normalize_clinical_structured_json_impl,
        normalize_support_level as _normalize_support_level_impl,
        parse_json_response as _parse_json_response_impl,
        render_structured_clinical as _render_structured_clinical_impl,
        render_structured_patient as _render_structured_patient_impl,
        resolve_cxr_anchor_decision as _resolve_cxr_anchor_decision_impl,
        safety_gate as _safety_gate_impl,
        sanitize_clinical_context as _sanitize_clinical_context_impl,
        build_cxr_anchor_section as _build_cxr_anchor_section_impl,
    )
    from .safety_audit import (
        REGION_DETECT_PROMPT as _REGION_DETECT_PROMPT,
        SAFETY_AUDIT_PROMPT as _SAFETY_AUDIT_PROMPT,
        check_vision_support as _check_vision_support_impl,
        compute_concordance as _compute_concordance_impl,
        detect_region_medgemma as _detect_region_medgemma_impl,
        run_safety_audit as _run_safety_audit_impl,
    )
    from .ui_cards import (
        make_reasoning_text as _make_reasoning_text_impl,
        make_safety_audit_card as _make_safety_audit_card_impl,
        make_triage_card as _make_triage_card_impl,
    )
except ImportError:
    from classifier import FractureClassifier
    from config import (
        CXR_MODEL_PATH,
        CXR_SOFT_LOCK_MARGIN,
        LINEAR_PROBE_PATH,
        MEDGEMMA_MODEL,
        OLLAMA_BASE_URL,
        SAFETY_AUDIT_ENABLED,
        SAFETY_AUDIT_TIMEOUT,
        TRIAGE_THRESHOLDS,
        VALIDATED_REGIONS,
    )
    from evidence import (
        get_model_transparency_info as _get_model_transparency_info,
        get_performance_context as _get_performance_context,
    )
    from prediction_logger import prediction_logger
    from reporting import (
        CLINICAL_JSON_PROMPT as _CLINICAL_JSON_PROMPT,
        CLINICAL_PROMPT as _CLINICAL_PROMPT,
        PATIENT_JSON_PROMPT as _PATIENT_JSON_PROMPT,
        PATIENT_PROMPT as _PATIENT_PROMPT,
        SAFETY_DISCLAIMER as _SAFETY_DISCLAIMER,
        STRUCTURED_JSON_FIELDS_CLINICAL as _STRUCTURED_JSON_FIELDS_CLINICAL,
        STRUCTURED_JSON_FIELDS_PATIENT as _STRUCTURED_JSON_FIELDS_PATIENT,
        append_metadata_block as _append_metadata_block_impl,
        build_audit_findings_section as _build_audit_findings_section_impl,
        build_clinical_summary_section as _build_clinical_summary_section_impl,
        build_context_section as _build_context_section_impl,
        clean_medgemma_output as _clean_medgemma_output_impl,
        generate_report as _generate_report_impl,
        normalize_clinical_structured_json as _normalize_clinical_structured_json_impl,
        normalize_support_level as _normalize_support_level_impl,
        parse_json_response as _parse_json_response_impl,
        render_structured_clinical as _render_structured_clinical_impl,
        render_structured_patient as _render_structured_patient_impl,
        resolve_cxr_anchor_decision as _resolve_cxr_anchor_decision_impl,
        safety_gate as _safety_gate_impl,
        sanitize_clinical_context as _sanitize_clinical_context_impl,
        build_cxr_anchor_section as _build_cxr_anchor_section_impl,
    )
    from safety_audit import (
        REGION_DETECT_PROMPT as _REGION_DETECT_PROMPT,
        SAFETY_AUDIT_PROMPT as _SAFETY_AUDIT_PROMPT,
        check_vision_support as _check_vision_support_impl,
        compute_concordance as _compute_concordance_impl,
        detect_region_medgemma as _detect_region_medgemma_impl,
        run_safety_audit as _run_safety_audit_impl,
    )
    from ui_cards import (
        make_reasoning_text as _make_reasoning_text_impl,
        make_safety_audit_card as _make_safety_audit_card_impl,
        make_triage_card as _make_triage_card_impl,
    )


# ---------------------------------------------------------------------------
# Report and audit constants (re-exported from split modules)
# ---------------------------------------------------------------------------

SAFETY_DISCLAIMER = _SAFETY_DISCLAIMER
CLINICAL_PROMPT = _CLINICAL_PROMPT
PATIENT_PROMPT = _PATIENT_PROMPT
CLINICAL_JSON_PROMPT = _CLINICAL_JSON_PROMPT
PATIENT_JSON_PROMPT = _PATIENT_JSON_PROMPT
STRUCTURED_JSON_FIELDS_CLINICAL = _STRUCTURED_JSON_FIELDS_CLINICAL
STRUCTURED_JSON_FIELDS_PATIENT = _STRUCTURED_JSON_FIELDS_PATIENT
SAFETY_AUDIT_PROMPT = _SAFETY_AUDIT_PROMPT
REGION_DETECT_PROMPT = _REGION_DETECT_PROMPT


def safety_gate(text: str) -> str:
    return _safety_gate_impl(text)


def append_metadata_block(text: str, result: dict) -> str:
    return _append_metadata_block_impl(text, result)


def _sanitize_clinical_context(text: str) -> str:
    return _sanitize_clinical_context_impl(text)


def _build_context_section(clinical_context: str) -> str:
    return _build_context_section_impl(clinical_context)


def _build_audit_findings_section(audit_result: dict | None) -> str:
    return _build_audit_findings_section_impl(audit_result)


def _build_clinical_summary_section(clinical_report: dict | None) -> str:
    return _build_clinical_summary_section_impl(clinical_report)


def _parse_json_response(raw: str) -> dict | None:
    return _parse_json_response_impl(raw)


def _normalize_support_level(value: str) -> str:
    return _normalize_support_level_impl(value)


def _resolve_cxr_anchor_decision(
    classification_result: dict,
    audit_result: dict | None,
    threshold_margin: float = CXR_SOFT_LOCK_MARGIN,
) -> dict:
    return _resolve_cxr_anchor_decision_impl(
        classification_result,
        audit_result,
        threshold_margin=threshold_margin,
    )


def _build_cxr_anchor_section(
    classification_result: dict,
    anchor_context: dict | None,
    audit_result: dict | None = None,
) -> str:
    return _build_cxr_anchor_section_impl(
        classification_result,
        anchor_context,
        audit_result,
    )


def _normalize_clinical_structured_json(
    parsed: dict,
    anchor_context: dict | None = None,
) -> dict:
    return _normalize_clinical_structured_json_impl(
        parsed,
        anchor_context=anchor_context,
    )


def _render_structured_clinical(parsed: dict) -> str:
    return _render_structured_clinical_impl(parsed)


def _render_structured_patient(parsed: dict) -> str:
    return _render_structured_patient_impl(parsed)


def _call_medgemma(prompt: str) -> dict:
    """Send a prompt to MedGemma via Ollama and return the raw API response dict.

    Raises requests.ConnectionError, requests.Timeout, or other exceptions
    on failure — callers handle these.
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MEDGEMMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
                "top_p": 0.9,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _call_medgemma_vision(prompt: str, image: Image.Image) -> dict:
    """Send a prompt + image to MedGemma via Ollama multimodal API.

    Converts the PIL image to grayscale, resizes to max 1024px, and
    encodes as base64 PNG for the Ollama /api/generate endpoint.
    """
    import base64

    # Preprocess: grayscale, resize to max 1024px
    img = image.convert("L")
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Encode to base64 PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MEDGEMMA_MODEL,
            "prompt": prompt,
            "images": [b64_string],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
                "top_p": 0.9,
            },
        },
        timeout=SAFETY_AUDIT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _detect_region_medgemma(image: Image.Image) -> tuple[str, bool]:
    return _detect_region_medgemma_impl(
        image,
        is_vision_available=is_vision_available,
        call_medgemma_vision=_call_medgemma_vision,
        parse_json_response=_parse_json_response,
    )


def generate_report(
    classification_result: dict,
    clinical_context: str = "",
    report_type: str = "clinical",
    audit_result: dict | None = None,
    clinical_summary: dict | None = None,
) -> dict:
    return _generate_report_impl(
        classification_result,
        clinical_context=clinical_context,
        report_type=report_type,
        audit_result=audit_result,
        clinical_summary=clinical_summary,
        call_medgemma=_call_medgemma,
    )


def _clean_medgemma_output(text: str) -> str:
    return _clean_medgemma_output_impl(text)


# ---------------------------------------------------------------------------
# Safety Audit — MedGemma Vision Cross-Validation
# ---------------------------------------------------------------------------

_vision_available = None
_vision_lock = threading.Lock()


def _check_vision_support() -> bool:
    return _check_vision_support_impl(OLLAMA_BASE_URL, MEDGEMMA_MODEL)


def is_vision_available() -> bool:
    """Cached, thread-safe check for MedGemma vision model availability."""
    global _vision_available
    if _vision_available is None:
        with _vision_lock:
            if _vision_available is None:
                _vision_available = _check_vision_support()
    return _vision_available


def _compute_concordance(cxr_probability: float, visual_assessment: str) -> dict:
    return _compute_concordance_impl(
        cxr_probability,
        visual_assessment,
        TRIAGE_THRESHOLDS,
    )


def run_safety_audit(image: Image.Image, classification_result: dict) -> dict:
    return _run_safety_audit_impl(
        image,
        classification_result,
        safety_audit_enabled=SAFETY_AUDIT_ENABLED,
        is_vision_available=is_vision_available,
        medgemma_model=MEDGEMMA_MODEL,
        call_medgemma_vision=_call_medgemma_vision,
        parse_json_response=_parse_json_response,
        normalize_support_level=_normalize_support_level,
        compute_concordance_fn=_compute_concordance,
        timeout_s=SAFETY_AUDIT_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Evidence / Transparency helpers
# ---------------------------------------------------------------------------

def get_model_transparency_info(result: dict) -> str:
    return _get_model_transparency_info(result)


def get_performance_context() -> str:
    return _get_performance_context()


# ---------------------------------------------------------------------------
# Batch Triage
# ---------------------------------------------------------------------------

def run_batch_triage(files):
    """Process multiple X-rays and return a prioritized triage summary table."""
    if not files:
        return "Upload one or more X-ray images to run batch triage."

    rows = []
    for f in files:
        filepath = f.name if hasattr(f, "name") else str(f)
        filename = Path(filepath).name
        try:
            img = Image.open(filepath)
            body_region, _detected = _detect_region_medgemma(img)
            result = get_classifier().classify(img, body_region)
            result["region_auto_detected"] = _detected

            # FIX #15: audit-log each batch prediction
            prediction_logger.log_prediction(
                result, body_region_input="auto",
            )

            color_emoji = {"red": "🔴", "yellow": "🟡", "green": "🟢"}[
                result["triage_color"]
            ]
            rows.append({
                "": color_emoji,
                "File": filename,
                "Fracture %": f"{round(result['probability'] * 100, 1)}%",
                "Triage": result["triage_level"],
                "Region": result["body_region"],
                "Latency": f"{result['latency_ms']}ms",
            })
        except Exception as e:
            rows.append({
                "": "⚠️",
                "File": filename,
                "Fracture %": "Error",
                "Triage": str(e)[:50],
                "Region": "Unknown",
                "Latency": "-",
            })

    # FIX #1: Sort numerically, not lexicographically. String sort would rank
    # "9.1%" above "78.5%" because "9" > "7" in ASCII — dangerous for triage.
    # Errors (non-numeric values) sort to the end with -1 sentinel.
    rows.sort(
        key=lambda r: float(r["Fracture %"].rstrip("%"))
        if r["Fracture %"] not in ("Error", "-")
        else -1,
        reverse=True,
    )

    if not rows:
        return "No images processed."

    headers = list(rows[0].keys())
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"

    summary_red = sum(1 for r in rows if r[""] == "🔴")
    summary_yellow = sum(1 for r in rows if r[""] == "🟡")
    summary_green = sum(1 for r in rows if r[""] == "🟢")

    md += f"\n**Summary:** {len(rows)} images processed — "
    md += f"🔴 {summary_red} high · 🟡 {summary_yellow} moderate · 🟢 {summary_green} low"

    return md


# ---------------------------------------------------------------------------
# UI State Management
# ---------------------------------------------------------------------------

_classifier = None
_classifier_lock = threading.Lock()


def get_classifier() -> FractureClassifier:
    """Thread-safe lazy singleton for the classifier."""
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:  # double-checked locking
                _classifier = FractureClassifier()
    return _classifier


def check_ollama_status():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.ok:
            models = [m["name"] for m in resp.json().get("models", [])]
            has_medgemma = any("medgemma" in m.lower() for m in models)
            if has_medgemma:
                status = "✅ Ollama running · MedGemma available"
            else:
                status = f"⚠️ Ollama running · MedGemma not found. Models: {', '.join(models) or 'none'}"
            # Safety audit status
            if SAFETY_AUDIT_ENABLED:
                status += f" · Safety audit: {'ready' if has_medgemma else 'model not found'}"
            else:
                status += " · Safety audit: disabled"
            return status
        return "⚠️ Ollama error"
    except requests.ConnectionError:
        return "❌ Ollama not running — start with: ollama serve"
    except Exception as e:
        return f"❌ {e}"


def step_triage(image, clinical_context):
    """Step 1 → Step 2 + Step 3: Run triage and auto-generate both reports."""
    if image is None:
        return (
            gr.update(visible=False), "", "",
            gr.update(visible=False), "",
            gr.update(visible=False), None,
            "", "", "", "",
        )

    pil_image = (
        Image.fromarray(image) if not isinstance(image, Image.Image) else image
    )
    body_region, region_detected = _detect_region_medgemma(pil_image)
    result = get_classifier().classify(pil_image, body_region)
    result["region_auto_detected"] = region_detected
    get_classifier().release_model()

    # FIX #15: audit-log every triage prediction
    prediction_logger.log_prediction(
        result,
        body_region_input="auto",
        clinical_context=clinical_context or "",
    )

    state = {
        "result": result,
        "clinical_context": clinical_context or "",
    }

    triage_html = _make_triage_card(result)
    reasoning = _make_reasoning_text(result)

    # Safety audit: MedGemma vision cross-validation
    # Runs after CXR classification, before report generation (sequential
    # to avoid Ollama GPU memory contention on Jetson)
    audit_result = run_safety_audit(pil_image, result)
    audit_html = _make_safety_audit_card(audit_result)
    state["audit_result"] = audit_result

    # Auto-generate both reports (chained: audit → clinical → patient)
    clinical = generate_report(
        result, clinical_context or "", report_type="clinical",
        audit_result=audit_result,
    )
    clinical_narrative, clinical_json = _format_report_output(clinical)

    patient = generate_report(
        result, clinical_context or "", report_type="patient",
        clinical_summary=clinical,
    )
    patient_narrative, patient_json = _format_report_output(patient)

    return (
        gr.update(visible=True),
        triage_html,
        reasoning,
        gr.update(visible=True),
        audit_html,
        gr.update(visible=True),
        state,
        clinical_narrative,
        clinical_json,
        patient_narrative,
        patient_json,
    )


def _format_report_output(report: dict) -> tuple[str, str]:
    """Format a report dict into (narrative_markdown, raw_json_block).

    Returns a 2-tuple so the UI can render the narrative prominently and
    the raw JSON in a collapsible accordion.
    """
    if report["error"]:
        return f"⚠️ {report['error']}", ""

    header = ""
    if report.get("report_type") == "clinical":
        header = "**CXR-Anchored Clinical Report**\n\n"

    narrative = header + report["report_text"] + (
        f"\n\n---\n*Generated by MedGemma 1.5 4B in {report['latency_s']}s*"
    )

    if report.get("structured_json"):
        raw_json = (
            "```json\n"
            + json.dumps(report["structured_json"], indent=2)
            + "\n```"
        )
    else:
        raw_json = "*Structured output unavailable — narrative fallback used.*"

    return narrative, raw_json


def step_clinical_report(state):
    """Generate clinician-facing report."""
    if not state or "result" not in state:
        return "*Run triage first.*", ""

    report = generate_report(
        state["result"], state.get("clinical_context", ""), report_type="clinical"
    )
    return _format_report_output(report)


def step_patient_report(state):
    """Generate patient-friendly report."""
    if not state or "result" not in state:
        return "*Run triage first.*", ""

    report = generate_report(
        state["result"], state.get("clinical_context", ""), report_type="patient"
    )
    return _format_report_output(report)


def step_show_evidence(state):
    """Load the evidence / transparency panel."""
    if not state or "result" not in state:
        return "*Run triage first.*", "*Run triage first.*"

    transparency = get_model_transparency_info(state["result"])
    performance = get_performance_context()
    return transparency, performance


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _make_triage_card(result: dict) -> str:
    return _make_triage_card_impl(result)


def _make_reasoning_text(result: dict) -> str:
    return _make_reasoning_text_impl(result)


def _make_safety_audit_card(audit_result: dict) -> str:
    return _make_safety_audit_card_impl(audit_result)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.disclaimer-box {
    background: #2d1b1b; border: 1px solid #5a2d2d; border-radius: 8px;
    padding: 12px 16px; color: #e8a0a0; font-size: 13px; margin-top: 8px;
}
footer { display: none !important; }
"""


def build_ui():
    with gr.Blocks(
        title="EdgeFracture — Fracture Triage",
    ) as app:

        triage_state = gr.State(value=None)

        # ── Header ──
        gr.Markdown(
            """
            # 🦴 EdgeFracture
            **Portable Musculoskeletal Fracture Triage on Edge Hardware**
            *CXR Foundation + MedGemma · Fully offline on Jetson Orin Nano ($249)*
            """
        )

        with gr.Tabs():
            # ==============================================================
            # TAB 1: Single Image Triage (Stepped)
            # ==============================================================
            with gr.Tab("🔍 Triage", id="triage"):

                # ── STEP 1: Upload & Context ──
                gr.Markdown("#### Step 1 — Upload X-ray & Provide Context")
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="X-ray Image", type="pil", height=350,
                        )
                    with gr.Column(scale=1):
                        clinical_context = gr.Textbox(
                            label="Clinical Context (optional)",
                            placeholder=(
                                "e.g., 'Patient fell on outstretched hand, "
                                "point tenderness over anatomical snuffbox, 45yo female'"
                            ),
                            lines=3,
                        )
                        triage_btn = gr.Button(
                            "🔍 Analyze X-ray", variant="primary", size="lg",
                        )

                # ── STEP 2: Triage Result ──
                step2_container = gr.Column(visible=False)
                with step2_container:
                    gr.Markdown("---")
                    gr.Markdown("#### Step 2 — Triage Result")
                    with gr.Row():
                        with gr.Column(scale=1):
                            triage_card = gr.HTML()
                        with gr.Column(scale=1):
                            reasoning_text = gr.Markdown()

                # ── CROSS-CHECK: MedGemma Visual Safety Audit ──
                audit_container = gr.Column(visible=False)
                with audit_container:
                    gr.Markdown("---")
                    gr.Markdown(
                        "#### Cross-Check — MedGemma Visual Safety Audit"
                    )
                    audit_card = gr.HTML()

                # ── STEP 3: Clinical Reports ──
                step3_container = gr.Column(visible=False)
                with step3_container:
                    gr.Markdown("---")
                    gr.Markdown("#### Step 3 — AI Clinical Report")
                    gr.Markdown(
                        "*MedGemma automatically generates reports for clinicians "
                        "and patients after triage. Clinical context (if provided "
                        "above) is included in the prompt.*"
                    )

                    with gr.Tabs():
                        with gr.Tab("🩺 Clinician View"):
                            clinical_report_md = gr.Markdown(
                                "*Report will appear after triage analysis.*"
                            )
                            with gr.Accordion(
                                "📦 Structured JSON (machine-readable)", open=False,
                            ):
                                clinical_json_md = gr.Markdown("")
                        with gr.Tab("💬 Patient View"):
                            patient_report_md = gr.Markdown(
                                "*Report will appear after triage analysis.*"
                            )
                            with gr.Accordion(
                                "📦 Structured JSON (machine-readable)", open=False,
                            ):
                                patient_json_md = gr.Markdown("")

                # ── STEP 4: Evidence & Transparency ──
                gr.Markdown("---")
                with gr.Accordion(
                    "📊 Evidence & Transparency — How It Works", open=False,
                ):
                    evidence_btn = gr.Button(
                        "Load Evidence Panel", variant="secondary", size="sm",
                    )
                    with gr.Tabs():
                        with gr.Tab("🔬 This Result"):
                            evidence_this_md = gr.Markdown(
                                "*Click 'Load Evidence Panel' to see how "
                                "this result was produced.*"
                            )
                        with gr.Tab("📈 Model Performance"):
                            evidence_perf_md = gr.Markdown(
                                "*Click 'Load Evidence Panel' to see "
                                "model benchmarks and data efficiency results.*"
                            )

                # ── Disclaimer ──
                gr.HTML("""
                <div class="disclaimer-box">
                    <strong>⚠️ Research Prototype — Not for Clinical Use</strong><br>
                    EdgeFracture is an AI-assisted screening tool for research and
                    demonstration only. Not FDA-cleared. Not a diagnostic device.
                    All outputs require review by a qualified healthcare professional.
                    AI screening supplements but never replaces clinical judgment.
                </div>
                """)

            # ==============================================================
            # TAB 2: Batch Triage
            # ==============================================================
            with gr.Tab("📋 Batch Triage", id="batch"):
                gr.Markdown(
                    """
                    #### Batch Triage Queue
                    Upload multiple X-ray images to get a prioritized triage
                    summary. Cases are **sorted by risk score** (highest first)
                    — simulating a real ED workflow where the most urgent cases
                    surface to the top.
                    """
                )
                batch_files = gr.File(
                    label="Upload X-ray Images",
                    file_count="multiple",
                    file_types=["image"],
                )
                batch_btn = gr.Button(
                    "🔍 Run Batch Triage", variant="primary", size="lg",
                )
                batch_results = gr.Markdown(
                    "*Upload images and click Run Batch Triage.*"
                )

            # ==============================================================
            # TAB 3: About / System
            # ==============================================================
            with gr.Tab("ℹ️ About", id="about"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """
                            ### EdgeFracture

                            **A portable, fully offline fracture screening station**
                            that repurposes a chest X-ray foundation model to triage
                            musculoskeletal fractures across body regions — running on
                            a $249 Jetson Orin Nano.

                            ---

                            **The Problem:** Missed fractures are the #1 diagnostic
                            error in emergency departments (3–10% miss rate).
                            4.3 billion people live in regions with fewer than
                            1 radiologist per 100,000 population.

                            **Our Approach:** Google's
                            [CXR Foundation](https://huggingface.co/google/cxr-foundation),
                            trained only on chest X-rays, transfers to musculoskeletal
                            fracture detection across 4 body regions — with as few as
                            500 labeled examples reaching 0.82 AUC.
                            [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it)
                            adds automatic body-region detection, visual safety
                            cross-checking, and clinician/patient communication.

                            **Key Result:** Overall AUC **0.882** across hand, leg,
                            hip, and shoulder — from a model that has never seen a
                            musculoskeletal X-ray during training.

                            ---

                            **HAI-DEF Models Used:**
                            - **CXR Foundation** — Image embeddings & fracture classification
                            - **MedGemma 1.5 4B** — Region detection, visual safety audit, and report generation

                            **Dataset:**
                            [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012)
                            — 4,083 MSK X-rays · CC BY 4.0

                            **Hardware:** NVIDIA Jetson Orin Nano 8GB · $249 · Fully offline

                            **Competition:**
                            [MedGemma Impact Challenge 2026](https://www.kaggle.com/competitions/medgemma-impact-challenge)
                            """
                        )
                    with gr.Column():
                        gr.Markdown("### System Status")
                        status_text = gr.Textbox(
                            value=check_ollama_status(),
                            label="Ollama / MedGemma",
                            interactive=False,
                        )
                        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

                        gr.Markdown(
                            f"""
                            ### Configuration
                            - **Ollama URL:** `{OLLAMA_BASE_URL}`
                            - **MedGemma:** `{MEDGEMMA_MODEL}`
                            - **CXR Foundation:** `{'Loaded ✓' if get_classifier().model_loaded else 'Not loaded — placeholder mode'}`
                            - **Linear probe:** `{'Loaded ✓ (0.882 AUC)' if get_classifier().probe_loaded else 'Not loaded — placeholder mode'}`
                            - **Region detection:** `MedGemma vision (auto)`
                            - **Safety Audit:** `{'Enabled ✓ — MedGemma vision cross-check' if SAFETY_AUDIT_ENABLED else 'Disabled'}`
                            """
                        )

        # ── Event wiring ──

        triage_btn.click(
            fn=step_triage,
            inputs=[image_input, clinical_context],
            outputs=[
                step2_container, triage_card, reasoning_text,
                audit_container, audit_card,
                step3_container, triage_state,
                clinical_report_md, clinical_json_md,
                patient_report_md, patient_json_md,
            ],
        )

        evidence_btn.click(
            fn=step_show_evidence,
            inputs=[triage_state],
            outputs=[evidence_this_md, evidence_perf_md],
        )

        batch_btn.click(
            fn=run_batch_triage,
            inputs=[batch_files],
            outputs=[batch_results],
        )

        refresh_btn.click(
            fn=check_ollama_status,
            inputs=[],
            outputs=[status_text],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFracture Triage App")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  EdgeFracture — Portable Fracture Triage")
    print("=" * 60)
    print(f"  Ollama:       {OLLAMA_BASE_URL}")
    print(f"  MedGemma:     {MEDGEMMA_MODEL}")
    print(f"  CXR Model:    {CXR_MODEL_PATH or 'Not configured'}")
    print(f"  Linear Probe: {LINEAR_PROBE_PATH or 'Not configured'}")
    print(f"  CXR loaded:   {'✓' if get_classifier().model_loaded else '✗ (placeholder mode)'}")
    print(f"  Probe loaded: {'✓' if get_classifier().probe_loaded else '✗ (placeholder mode)'}")
    print(f"  Region:       MedGemma vision (auto-detect)")
    print("=" * 60)

    app = build_ui()
    app.launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )
