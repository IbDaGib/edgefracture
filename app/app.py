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
import re
import threading
import time
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
    from ui_cards import (
        make_reasoning_text as _make_reasoning_text_impl,
        make_safety_audit_card as _make_safety_audit_card_impl,
        make_triage_card as _make_triage_card_impl,
    )


# ---------------------------------------------------------------------------
# Safety Gate — programmatic safety disclaimers
# ---------------------------------------------------------------------------

_OVERCONFIDENT_PATTERNS = [
    r"\bdiagnos(?:is|ed|e)\b",
    r"\bconfirm(?:s|ed)?\b",
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bclearly shows\b",
    r"\bI can see\b",
    r"\bthe fracture is\b",
    r"\bthis is a\b.*\bfracture\b",
]

SAFETY_DISCLAIMER = (
    "\n\n---\n⚠️ **AI-assisted screening — requires clinician confirmation.** "
    "This is a research prototype, not a diagnostic device. All findings must be "
    "reviewed by a qualified healthcare professional."
)


def safety_gate(text: str) -> str:
    """Scan MedGemma output for overconfident language and append safety disclaimer."""
    # Always append the disclaimer
    text = text.rstrip()

    # Flag if overconfident patterns found
    for pattern in _OVERCONFIDENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(
                pattern,
                lambda m: f"*{m.group(0)}*",  # italicize overconfident terms
                text,
                flags=re.IGNORECASE,
            )

    if not text.endswith(SAFETY_DISCLAIMER.strip()):
        text += SAFETY_DISCLAIMER

    return text


def append_metadata_block(text: str, result: dict) -> str:
    """Programmatically append structured metadata — never rely on MedGemma for this."""
    block = (
        f"\n\n**Screening Metadata**\n"
        f"- CXR Foundation confidence: {round(result['probability'] * 100, 1)}%\n"
        f"- Body region: {result['body_region']}\n"
        f"- Triage level: {result['triage_level']}\n"
        f"- Model: CXR Foundation ELIXR-C v2 → Linear Probe\n"
        f"- Report: MedGemma 1.5 4B (Q4_K_M)\n"
        f"- Inference: {result['latency_ms']}ms (classification)"
    )
    return text + block


# ---------------------------------------------------------------------------
# MedGemma Report Generator
# ---------------------------------------------------------------------------

CLINICAL_PROMPT = """\
You are a musculoskeletal radiology assistant. An AI screening system \
has analyzed a {body_region} X-ray and produced the following results:

- Fracture probability: {score_pct}% ({triage_level})
- Classification: {classification}
- Confidence: {confidence}
{anchor_context_section}{clinical_context_section}{audit_findings_section}
Provide a CLINICAL SUMMARY for the reviewing physician:

1. Provide a CXR-anchored impression (2-3 sentences). Use CXR probability as \
the primary anchor and use visual findings to support/temper confidence.

2. Include specific diagnostic detail:
   - most likely anatomic location of injury
   - likely fracture pattern/type for this body region

3. URGENCY LEVEL — classify as one of:
   - URGENT: Likely displaced or unstable fracture — splint and refer immediately
   - MODERATE: Possible non-displaced fracture — further imaging recommended
   - LOW: Low suspicion — clinical correlation advised

4. Provide urgency rationale and recommended next steps \
(imaging, referral, follow-up).

Be concise and clinical. This is an AI screening result, not a diagnosis."""

PATIENT_PROMPT = """\
You are a friendly medical assistant explaining an X-ray screening result \
to a patient. The AI screening found:

- Body part: {body_region}
- Result: {classification} ({score_pct}% probability)
- Urgency: {triage_level}
{clinical_context_section}{clinical_summary_section}
Explain this result in 3-4 sentences using simple, reassuring language. \
Avoid medical jargon. Let them know what happens next and that a doctor \
will review everything. Do not be alarming — be honest but calming."""


# -- Structured JSON prompts ------------------------------------------------

CLINICAL_JSON_PROMPT = """\
You are a musculoskeletal radiology assistant. An AI screening system \
has analyzed a {body_region} X-ray and produced the following results:

- Fracture probability: {score_pct}% ({triage_level})
- Classification: {classification}
- Confidence: {confidence}
{anchor_context_section}{clinical_context_section}
Use CXR probability/classification as the primary anchor. If visual findings are discordant \
or image quality is limited, keep the CXR-anchored impression but temper confidence/urgency rationale.

Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) using this exact schema:

{{"anchor_impression": "1-2 sentence CXR-anchored clinical impression",\
 "anatomic_location": "Most likely anatomic injury location",\
 "likely_fracture_pattern": "Most likely fracture pattern/type for this region",\
 "urgency": "URGENT | MODERATE | LOW",\
 "urgency_rationale": "Why this urgency follows from CXR anchor plus visual support/limitations",\
 "recommended_imaging": "Specific imaging recommendation",\
 "immediate_actions": "Immediate clinical actions",\
 "confidence_rationale": "Why confidence is high/moderate/tempered",\
 "discordance_note": "Brief note on concordance/discordance or image limitations"}}"""

PATIENT_JSON_PROMPT = """\
You are a friendly medical assistant explaining an X-ray screening result \
to a patient. The AI screening found:

- Body part: {body_region}
- Result: {classification} ({score_pct}% probability)
- Urgency: {triage_level}
{clinical_context_section}{clinical_summary_section}
Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) using this exact schema:

{{"summary": "2-3 sentence plain-language explanation of the result",\
 "next_steps": "What happens next, in simple terms",\
 "reassurance": "A brief reassuring statement about the process"}}"""

SAFETY_AUDIT_PROMPT = """\
You are a musculoskeletal radiology assistant performing a visual safety audit \
of an X-ray image. A separate AI system (CXR Foundation) has already provided \
a numeric fracture probability. Use that CXR output as the primary anchor, then \
evaluate whether the image visually supports or limits that anchor.

Body region: {body_region}
CXR fracture probability anchor: {cxr_probability_pct}%
CXR triage anchor: {cxr_triage_level}
CXR classification anchor: {cxr_classification}

Analyze this X-ray image and look for:
- Cortical breaks or discontinuities
- Bone displacement or angulation
- Soft tissue swelling adjacent to bone
- Joint space irregularities
- Periosteal reaction or callus formation

Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) \
using this exact schema:

{{"visual_assessment": "fracture_likely | fracture_unlikely | uncertain",\
 "support_level": "supports_cxr | partially_supports_cxr | does_not_support_cxr | image_limited",\
 "suspected_location": "specific anatomic location if visible, else Unspecified",\
 "suspected_pattern": "likely fracture pattern if visible, else Unspecified",\
 "observations": ["list of specific findings observed in the image"],\
 "confidence_note": "brief note on image quality or limitations",\
 "reasoning": "1-2 sentence explanation of your assessment"}}

IMPORTANT:
- Do NOT provide a numeric probability — that is CXR Foundation's job
- Only report what you can actually observe in the image
- If image quality is poor or findings are ambiguous, use "uncertain"
- Focus on observable findings, not clinical assumptions"""

REGION_DETECT_PROMPT = """\
You are a medical imaging assistant. Look at this X-ray image and identify \
the body region shown.

Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) \
using this exact schema:

{{"region": "<body region name>"}}

IMPORTANT:
- Respond with a single, specific anatomical region name (e.g., Hand, Wrist, Elbow, \
Forearm, Shoulder, Humerus, Spine, Ribs, Hip, Pelvis, Knee, Leg, Ankle, Foot)
- Use standard clinical anatomical terms
- Capitalize the first letter (e.g., "Elbow", not "elbow")
- Choose the MOST SPECIFIC region you can identify
- If you cannot determine the region, respond: {{"region": "Unknown"}}"""

STRUCTURED_JSON_FIELDS_CLINICAL = [
    "anchor_impression", "anatomic_location", "likely_fracture_pattern", "urgency",
    "urgency_rationale", "recommended_imaging", "immediate_actions",
    "confidence_rationale", "discordance_note",
]
STRUCTURED_JSON_FIELDS_PATIENT = ["summary", "next_steps", "reassurance"]


def _sanitize_clinical_context(text: str) -> str:
    """Sanitize user-provided clinical context before interpolation into LLM prompt.

    FIX #7: Prevents prompt injection by stripping instruction-like patterns,
    truncating to a safe length, and removing prompt-breaking characters.
    """
    # 1. Truncate to 500 characters max
    text = text.strip()[:500]

    # 2. Strip instruction-like patterns that could hijack the LLM prompt
    injection_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?previous",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s*prompt",
        r"override\s+(all\s+)?instructions",
        r"forget\s+(all\s+)?(previous|above|prior)",
        r"act\s+as\s+(a\s+)?",
        r"pretend\s+(you\s+are|to\s+be)",
        r"do\s+not\s+follow",
        r"instead[,:]?\s+(you\s+should|do|say|write)",
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, "[removed]", text, flags=re.IGNORECASE)

    # 3. Remove characters that could break prompt formatting
    #    (keep alphanumeric, basic punctuation, and medical symbols)
    text = re.sub(r"[{}<>|\\`]", "", text)

    return text.strip()


def _build_context_section(clinical_context: str) -> str:
    if clinical_context and clinical_context.strip():
        sanitized = _sanitize_clinical_context(clinical_context)
        if sanitized:
            return f"\nClinical context provided: {sanitized}\n"
    return ""


def _build_audit_findings_section(audit_result: dict | None) -> str:
    """Format safety audit findings into a prompt section for clinical reports.

    Returns empty string if audit was skipped, errored, or has no useful data.
    """
    if not audit_result or audit_result.get("skipped"):
        return ""

    visual = audit_result.get("visual_assessment", "")
    concordance = audit_result.get("concordance", "")
    reasoning = audit_result.get("reasoning", "")
    observations = audit_result.get("observations", [])
    confidence_note = audit_result.get("confidence_note", "")
    support_level = audit_result.get("support_level", "")
    suspected_location = audit_result.get("suspected_location", "")
    suspected_pattern = audit_result.get("suspected_pattern", "")

    # If there's nothing useful, return empty
    if not visual and not observations:
        return ""

    lines = ["\nMedGemma Visual Safety Audit (advisory context — not directive):"]
    if visual:
        lines.append(f"- Visual assessment: {visual}")
    if concordance and reasoning:
        lines.append(f"- Cross-check result: {concordance} — {reasoning}")
    if support_level:
        lines.append(f"- Support level: {support_level}")
    if suspected_location:
        lines.append(f"- Suspected location: {suspected_location}")
    if suspected_pattern:
        lines.append(f"- Suspected pattern: {suspected_pattern}")
    if observations:
        lines.append("- Observations:")
        for obs in observations[:5]:
            lines.append(f"  - {obs}")
    if confidence_note:
        lines.append(f"- Image quality note: {confidence_note}")
    lines.append("")

    return "\n".join(lines)


def _build_clinical_summary_section(clinical_report: dict | None) -> str:
    """Format clinical report output into a prompt section for patient reports.

    Uses structured JSON fields when available, falls back to truncated
    narrative text. Returns empty string if clinical report errored.
    """
    if not clinical_report or clinical_report.get("error"):
        return ""

    parsed = clinical_report.get("structured_json")
    if parsed:
        parsed = _normalize_clinical_structured_json(parsed)
        lines = [
            "\nClinical AI Summary (use as the basis for your explanation — "
            "translate into patient-friendly language):"
        ]
        if parsed.get("primary_finding"):
            lines.append(f"- Finding: {parsed['primary_finding']}")
        if parsed.get("urgency"):
            lines.append(f"- Urgency: {parsed['urgency']}")
        if parsed.get("recommendation"):
            lines.append(f"- Recommendation: {parsed['recommendation']}")
        if parsed.get("differential"):
            lines.append(f"- Differential: {parsed['differential']}")
        if parsed.get("clinical_note"):
            lines.append(f"- Clinical note: {parsed['clinical_note']}")
        lines.append("")
        return "\n".join(lines)

    # Fallback: use narrative text (truncated)
    narrative = clinical_report.get("report_text", "").strip()
    if narrative:
        truncated = narrative[:500]
        if len(narrative) > 500:
            truncated += "..."
        return (
            f"\nClinical AI Summary (use as the basis for your explanation — "
            f"translate into patient-friendly language):\n{truncated}\n"
        )

    return ""


def _parse_json_response(raw: str) -> dict | None:
    """Try to extract a JSON object from MedGemma's response.

    Handles common LLM quirks: markdown code fences, trailing text, etc.
    Returns the parsed dict or None if parsing fails.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Try the whole string first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find a JSON object within the text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _normalize_support_level(value: str) -> str:
    """Normalize MedGemma support-level label to an allowed enum."""
    allowed = {
        "supports_cxr",
        "partially_supports_cxr",
        "does_not_support_cxr",
        "image_limited",
    }
    if not value:
        return "image_limited"
    normalized = str(value).strip().lower()
    return normalized if normalized in allowed else "image_limited"


def _resolve_cxr_anchor_decision(
    classification_result: dict,
    audit_result: dict | None,
    threshold_margin: float = CXR_SOFT_LOCK_MARGIN,
) -> dict:
    """Resolve final urgency using a CXR-led soft-lock policy."""
    prob = float(classification_result.get("probability", 0.5))

    if prob >= TRIAGE_THRESHOLDS["red"]:
        cxr_urgency = "URGENT"
    elif prob >= TRIAGE_THRESHOLDS["yellow"]:
        cxr_urgency = "MODERATE"
    else:
        cxr_urgency = "LOW"

    support_level = _normalize_support_level(
        (audit_result or {}).get("support_level", "")
    )
    concordance = (audit_result or {}).get("concordance", "UNCERTAIN")
    visual_assessment = str((audit_result or {}).get("visual_assessment", "")).strip().lower()
    confidence_note = str((audit_result or {}).get("confidence_note", "")).lower()

    image_limited = (
        support_level == "image_limited"
        or visual_assessment == "uncertain"
        or "poor" in confidence_note
        or "limited" in confidence_note
        or "low quality" in confidence_note
    )
    discordant = (
        concordance == "DISCORDANT"
        or support_level == "does_not_support_cxr"
    )
    tempered_confidence = discordant or image_limited or concordance == "UNCERTAIN"

    near_threshold = any(
        abs(prob - threshold) <= threshold_margin
        for threshold in TRIAGE_THRESHOLDS.values()
    )

    downgrade_map = {"URGENT": "MODERATE", "MODERATE": "LOW", "LOW": "LOW"}
    urgency_adjusted = tempered_confidence and near_threshold and cxr_urgency != "LOW"
    final_urgency = downgrade_map[cxr_urgency] if urgency_adjusted else cxr_urgency

    if discordant:
        discordance_note = (
            "Visual audit is discordant with the CXR anchor; confidence is tempered."
        )
    elif image_limited:
        discordance_note = (
            "Visual audit is image-limited/uncertain; confidence is tempered."
        )
    elif concordance == "UNCERTAIN":
        discordance_note = (
            "Cross-check is uncertain; maintain CXR anchor with tempered confidence."
        )
    else:
        discordance_note = "Visual audit supports the CXR anchor."

    return {
        "cxr_probability_pct": round(prob * 100, 1),
        "classification_anchor": (
            "Fracture detected" if prob >= 0.5 else "No fracture detected"
        ),
        "cxr_urgency": cxr_urgency,
        "final_urgency": final_urgency,
        "urgency_adjusted": urgency_adjusted,
        "tempered_confidence": tempered_confidence,
        "support_level": support_level,
        "concordance": concordance,
        "discordance_note": discordance_note,
    }


def _build_cxr_anchor_section(
    classification_result: dict,
    anchor_context: dict | None,
    audit_result: dict | None = None,
) -> str:
    """Build compact CXR anchor context for clinical prompts."""
    if not anchor_context:
        return ""

    location = (
        (audit_result or {}).get("suspected_location", "")
        or "Unspecified"
    )
    pattern = (
        (audit_result or {}).get("suspected_pattern", "")
        or "Unspecified"
    )

    lines = [
        "\nCXR Anchor Context (primary signal):",
        f"- Fracture probability anchor: {anchor_context['cxr_probability_pct']}%",
        f"- CXR triage level: {classification_result.get('triage_level', 'Unknown')}",
        f"- Classification anchor: {anchor_context['classification_anchor']}",
        f"- CXR urgency anchor: {anchor_context['cxr_urgency']}",
        f"- Final urgency policy output: {anchor_context['final_urgency']}",
        f"- Concordance: {anchor_context.get('concordance', 'UNCERTAIN')}",
        f"- Support level: {anchor_context.get('support_level', 'image_limited')}",
        f"- Suspected location: {location}",
        f"- Suspected pattern: {pattern}",
        f"- Discordance note: {anchor_context.get('discordance_note', '')}",
        "",
    ]
    return "\n".join(lines)


def _normalize_clinical_structured_json(
    parsed: dict,
    anchor_context: dict | None = None,
) -> dict:
    """Map new clinical schema fields to legacy keys for compatibility."""
    if not isinstance(parsed, dict):
        return {}

    normalized = dict(parsed)

    # Forward fill new fields from legacy payloads (if model returns old schema)
    if not normalized.get("anchor_impression") and normalized.get("primary_finding"):
        normalized["anchor_impression"] = normalized["primary_finding"]
    if not normalized.get("likely_fracture_pattern") and normalized.get("differential"):
        normalized["likely_fracture_pattern"] = normalized["differential"]
    if not normalized.get("immediate_actions") and normalized.get("recommendation"):
        normalized["immediate_actions"] = normalized["recommendation"]
    if not normalized.get("confidence_rationale") and normalized.get("clinical_note"):
        normalized["confidence_rationale"] = normalized["clinical_note"]
    if not normalized.get("urgency") and anchor_context:
        normalized["urgency"] = anchor_context.get("final_urgency", "")

    # Backward compatibility fields used by downstream chaining
    if not normalized.get("primary_finding") and normalized.get("anchor_impression"):
        normalized["primary_finding"] = normalized["anchor_impression"]
    if not normalized.get("differential") and normalized.get("likely_fracture_pattern"):
        normalized["differential"] = normalized["likely_fracture_pattern"]

    if not normalized.get("recommendation"):
        recommendation_parts = []
        if normalized.get("immediate_actions"):
            recommendation_parts.append(str(normalized["immediate_actions"]).strip())
        if normalized.get("recommended_imaging"):
            recommendation_parts.append(
                f"Imaging: {str(normalized['recommended_imaging']).strip()}"
            )
        recommendation = " ".join(p for p in recommendation_parts if p).strip()
        if recommendation:
            normalized["recommendation"] = recommendation

    if not normalized.get("clinical_note"):
        note_parts = []
        if normalized.get("confidence_rationale"):
            note_parts.append(str(normalized["confidence_rationale"]).strip())
        discordance_note = (
            normalized.get("discordance_note")
            or (anchor_context or {}).get("discordance_note", "")
        )
        if discordance_note:
            note_parts.append(str(discordance_note).strip())
        clinical_note = " ".join(p for p in note_parts if p).strip()
        if clinical_note:
            normalized["clinical_note"] = clinical_note

    return normalized


def _render_structured_clinical(parsed: dict) -> str:
    """Render parsed clinical JSON into a readable markdown report."""
    parsed = _normalize_clinical_structured_json(parsed)

    lines = []
    lines.append("**CXR-Anchored Clinical Synthesis**")

    if parsed.get("anchor_impression") or parsed.get("primary_finding"):
        lines.append(
            f"**Impression:** {parsed.get('anchor_impression') or parsed.get('primary_finding')}"
        )
    if parsed.get("anatomic_location"):
        lines.append(f"**Anatomic Location:** {parsed['anatomic_location']}")
    if parsed.get("likely_fracture_pattern") or parsed.get("differential"):
        lines.append(
            f"**Likely Pattern:** {parsed.get('likely_fracture_pattern') or parsed.get('differential')}"
        )
    if parsed.get("urgency"):
        urgency = parsed["urgency"].upper()
        icon = {"URGENT": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(urgency, "⚪")
        lines.append(f"**Urgency:** {icon} {urgency}")
    if parsed.get("urgency_rationale"):
        lines.append(f"**Urgency Rationale:** {parsed['urgency_rationale']}")
    if parsed.get("immediate_actions"):
        lines.append(f"**Immediate Actions:** {parsed['immediate_actions']}")
    if parsed.get("recommended_imaging"):
        lines.append(f"**Recommended Imaging:** {parsed['recommended_imaging']}")
    if parsed.get("confidence_rationale"):
        lines.append(f"**Confidence Rationale:** {parsed['confidence_rationale']}")
    if parsed.get("discordance_note"):
        lines.append(f"**Discordance Note:** {parsed['discordance_note']}")

    # Legacy fallback lines (if model emits old schema)
    if parsed.get("recommendation") and not parsed.get("immediate_actions"):
        lines.append(f"**Recommendation:** {parsed['recommendation']}")
    if parsed.get("clinical_note") and not parsed.get("confidence_rationale"):
        lines.append(f"**Note:** {parsed['clinical_note']}")

    rendered = "\n\n".join(line for line in lines if line)
    return rendered if rendered.strip() != "**CXR-Anchored Clinical Synthesis**" else ""


def _render_structured_patient(parsed: dict) -> str:
    """Render parsed patient JSON into a friendly markdown explanation."""
    lines = []
    if parsed.get("summary"):
        lines.append(parsed["summary"])
    if parsed.get("next_steps"):
        lines.append(f"**What happens next:** {parsed['next_steps']}")
    if parsed.get("reassurance"):
        lines.append(f"*{parsed['reassurance']}*")
    return "\n\n".join(lines)


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
    """Detect body region from an X-ray image using MedGemma vision.

    Returns (region_name, detected) where detected is True if MedGemma
    successfully identified the region. Falls back to ("Unknown", False)
    on any failure — never raises.
    """
    if not is_vision_available():
        return ("Unknown", False)

    try:
        data = _call_medgemma_vision(REGION_DETECT_PROMPT, image)
        raw = data.get("response", "").strip()

        # Try JSON parse first — accepts ANY region name
        parsed = _parse_json_response(raw)
        if parsed and parsed.get("region"):
            region = parsed["region"].strip().title()
            if region and region != "Unknown":
                return (region, True)

        # Fallback: string match common anatomical terms in free-text
        if raw:
            raw_lower = raw.lower()
            _COMMON_REGIONS = [
                "Hand", "Wrist", "Forearm", "Elbow", "Humerus", "Shoulder",
                "Clavicle", "Scapula", "Ribs", "Spine", "Pelvis", "Hip",
                "Femur", "Knee", "Leg", "Tibia", "Fibula", "Ankle", "Foot",
                "Skull", "Cervical", "Thoracic", "Lumbar", "Sacrum",
            ]
            for r in _COMMON_REGIONS:
                if r.lower() in raw_lower:
                    return (r, True)

        return ("Unknown", False)
    except Exception:
        return ("Unknown", False)


def generate_report(
    classification_result: dict,
    clinical_context: str = "",
    report_type: str = "clinical",
    audit_result: dict | None = None,
    clinical_summary: dict | None = None,
) -> dict:
    """Generate a structured + narrative report via MedGemma.

    Tries structured JSON output first. On parse failure, falls back to
    the narrative prompt. The returned dict always includes:
      - report_text: markdown narrative (safety-gated + metadata)
      - structured_json: parsed dict or None
      - latency_s: total wall-clock seconds
      - error: error string or None

    Optional chaining parameters:
      - audit_result: safety audit dict, injected into clinical prompts
      - clinical_summary: clinical report dict, injected into patient prompts
    """
    prob = classification_result["probability"]
    anchor_context = None
    if report_type == "clinical":
        anchor_context = _resolve_cxr_anchor_decision(
            classification_result, audit_result,
        )

    anchor_context_section = (
        _build_cxr_anchor_section(
            classification_result, anchor_context, audit_result,
        )
        if report_type == "clinical"
        else ""
    )
    context_section = _build_context_section(clinical_context)
    audit_section = _build_audit_findings_section(audit_result) if report_type == "clinical" else ""
    clinical_section = _build_clinical_summary_section(clinical_summary) if report_type == "patient" else ""

    # Base kwargs for JSON prompt — keep clean so MedGemma 4B reliably
    # follows the "respond with ONLY a valid JSON object" instruction.
    base_kwargs = dict(
        body_region=classification_result["body_region"],
        score_pct=round(prob * 100, 1),
        triage_level=classification_result["triage_level"],
        classification="Fracture detected" if prob >= 0.5 else "No fracture detected",
        confidence=classification_result["confidence"],
        anchor_context_section=anchor_context_section,
        clinical_context_section=context_section,
        audit_findings_section="",
        clinical_summary_section="",
    )

    # Enriched kwargs for narrative fallback — includes chained context
    narrative_kwargs = {
        **base_kwargs,
        "audit_findings_section": audit_section,
        "clinical_summary_section": clinical_section,
    }

    json_template = CLINICAL_JSON_PROMPT if report_type == "clinical" else PATIENT_JSON_PROMPT
    narrative_template = CLINICAL_PROMPT if report_type == "clinical" else PATIENT_PROMPT
    render_fn = _render_structured_clinical if report_type == "clinical" else _render_structured_patient

    start = time.time()
    try:
        # --- Attempt 1: structured JSON (clean prompt) ---
        json_prompt = json_template.format(**base_kwargs)
        data = _call_medgemma(json_prompt)
        raw_response = data.get("response", "").strip()
        parsed = _parse_json_response(raw_response)

        if parsed is not None:
            if report_type == "clinical":
                parsed = _normalize_clinical_structured_json(
                    parsed, anchor_context=anchor_context,
                )
            # Structured output succeeded — render into narrative
            report_text = render_fn(parsed)
        else:
            # JSON parse failed — fall back to enriched narrative prompt
            narrative_prompt = narrative_template.format(**narrative_kwargs)
            data = _call_medgemma(narrative_prompt)
            report_text = _clean_medgemma_output(data.get("response", "").strip())

        # Apply safety gate and metadata to the narrative
        report_text = safety_gate(report_text)
        report_text = append_metadata_block(report_text, classification_result)

        latency = time.time() - start
        return {
            "report_text": report_text,
            "structured_json": parsed,
            "latency_s": round(latency, 1),
            "error": None,
            "report_type": report_type,
            "anchor_context": anchor_context,
        }

    except requests.ConnectionError:
        return {
            "report_text": "",
            "structured_json": None,
            "latency_s": 0,
            "error": "Cannot connect to Ollama. Start with: ollama serve",
            "report_type": report_type,
            "anchor_context": anchor_context,
        }
    except requests.Timeout:
        return {
            "report_text": "",
            "structured_json": None,
            "latency_s": round(time.time() - start, 1),
            "error": "MedGemma timed out (>120s).",
            "report_type": report_type,
            "anchor_context": anchor_context,
        }
    except Exception as e:
        return {
            "report_text": "",
            "structured_json": None,
            "latency_s": round(time.time() - start, 1),
            "error": f"MedGemma error: {str(e)}",
            "report_type": report_type,
            "anchor_context": anchor_context,
        }


def _clean_medgemma_output(text: str) -> str:
    """Strip MedGemma artifacts: leaked prompts, critique loops, unused tokens."""
    text = re.sub(r"<unused\d+>", "", text)
    if "You are a musculoskeletal" in text or "You are a friendly" in text:
        for marker in [
            "1.", "CLINICAL SUMMARY", "Clinical Summary",
            "The AI", "Your X-ray", "Based on",
        ]:
            idx = text.find(marker)
            if idx > 0:
                text = text[idx:]
                break
    text = re.sub(r"\*\*Critique:?\*\*.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(
        r"\*\*Self-assessment:?\*\*.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ---------------------------------------------------------------------------
# Safety Audit — MedGemma Vision Cross-Validation
# ---------------------------------------------------------------------------

_vision_available = None
_vision_lock = threading.Lock()


def _check_vision_support() -> bool:
    """Query Ollama to check if the configured MedGemma model is available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.ok:
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(MEDGEMMA_MODEL in m or m in MEDGEMMA_MODEL for m in models)
        return False
    except Exception:
        return False


def is_vision_available() -> bool:
    """Cached, thread-safe check for MedGemma vision model availability."""
    global _vision_available
    if _vision_available is None:
        with _vision_lock:
            if _vision_available is None:
                _vision_available = _check_vision_support()
    return _vision_available


def _compute_concordance(cxr_probability: float, visual_assessment: str) -> dict:
    """Compare CXR Foundation probability with MedGemma visual assessment.

    Returns a dict with concordance status and reasoning.
    """
    # Map CXR probability to simplified category
    if cxr_probability >= TRIAGE_THRESHOLDS["red"]:  # >= 0.70
        cxr_category = "fracture_likely"
    elif cxr_probability < TRIAGE_THRESHOLDS["yellow"]:  # < 0.40
        cxr_category = "fracture_unlikely"
    else:
        cxr_category = "uncertain"

    # Normalize visual assessment
    va = visual_assessment.lower().strip() if visual_assessment else ""
    if va not in ("fracture_likely", "fracture_unlikely", "uncertain"):
        return {
            "concordance": "UNCERTAIN",
            "cxr_category": cxr_category,
            "visual_assessment": visual_assessment,
            "reasoning": (
                f"MedGemma returned unexpected assessment '{visual_assessment}'"
                " — defaulting to UNCERTAIN."
            ),
        }

    # Concordance matrix
    if cxr_category == va:
        concordance = "CONCORDANT"
        reasoning = (
            f"Both models agree: CXR Foundation ({cxr_category}) and "
            f"MedGemma Vision ({va}) reached the same conclusion."
        )
    elif cxr_category == "uncertain" or va == "uncertain":
        concordance = "UNCERTAIN"
        reasoning = (
            f"One or both models are uncertain: CXR Foundation ({cxr_category}), "
            f"MedGemma Vision ({va})."
        )
    else:
        concordance = "DISCORDANT"
        reasoning = (
            f"Models DISAGREE: CXR Foundation says {cxr_category} but "
            f"MedGemma Vision says {va}. Manual review recommended."
        )

    return {
        "concordance": concordance,
        "cxr_category": cxr_category,
        "visual_assessment": va,
        "reasoning": reasoning,
    }


def run_safety_audit(image: Image.Image, classification_result: dict) -> dict:
    """Run MedGemma vision analysis and compare with CXR Foundation result.

    Returns a dict with concordance, observations, and metadata.
    Always returns a valid dict — never raises.
    """
    if not SAFETY_AUDIT_ENABLED:
        return {
            "skipped": True,
            "reason": "Safety audit disabled via SAFETY_AUDIT_ENABLED=false",
        }

    if not is_vision_available():
        return {
            "skipped": True,
            "reason": (
                f"MedGemma model not available. "
                f"Pull it with: ollama pull {MEDGEMMA_MODEL}"
            ),
        }

    start = time.time()
    try:
        prob = float(classification_result.get("probability", 0.5))
        triage_level = classification_result.get("triage_level", "UNKNOWN")
        cxr_classification = (
            "Fracture detected" if prob >= 0.5 else "No fracture detected"
        )

        prompt = SAFETY_AUDIT_PROMPT.format(
            body_region=classification_result.get("body_region", "Unknown"),
            cxr_probability_pct=round(prob * 100, 1),
            cxr_triage_level=triage_level,
            cxr_classification=cxr_classification,
        )
        data = _call_medgemma_vision(prompt, image)
        raw_response = data.get("response", "").strip()

        parsed = _parse_json_response(raw_response)
        if parsed is None:
            visual_assessment = "uncertain"
            observations = []
            confidence_note = "MedGemma returned non-JSON response"
            reasoning = raw_response[:200] if raw_response else "No response"
            support_level = "image_limited"
            suspected_location = "Unspecified"
            suspected_pattern = "Unspecified"
        else:
            visual_assessment = parsed.get("visual_assessment", "uncertain")
            observations = parsed.get("observations", [])
            if not isinstance(observations, list):
                observations = []
            confidence_note = parsed.get("confidence_note", "")
            reasoning = parsed.get("reasoning", "")
            support_level = _normalize_support_level(parsed.get("support_level", ""))
            suspected_location = str(parsed.get("suspected_location", "")).strip() or "Unspecified"
            suspected_pattern = str(parsed.get("suspected_pattern", "")).strip() or "Unspecified"

        concordance_result = _compute_concordance(
            classification_result["probability"],
            visual_assessment,
        )

        latency = round(time.time() - start, 1)
        return {
            "skipped": False,
            "concordance": concordance_result["concordance"],
            "cxr_category": concordance_result["cxr_category"],
            "visual_assessment": visual_assessment,
            "observations": observations,
            "confidence_note": confidence_note,
            "support_level": support_level,
            "suspected_location": suspected_location,
            "suspected_pattern": suspected_pattern,
            "reasoning": concordance_result["reasoning"],
            "vision_reasoning": reasoning,
            "latency_s": latency,
            "error": None,
        }
    except requests.ConnectionError:
        return {
            "skipped": False,
            "concordance": "UNCERTAIN",
            "observations": [],
            "support_level": "image_limited",
            "suspected_location": "Unspecified",
            "suspected_pattern": "Unspecified",
            "reasoning": "Cannot connect to Ollama for vision analysis.",
            "latency_s": round(time.time() - start, 1),
            "error": "Connection refused — is Ollama running?",
        }
    except requests.Timeout:
        return {
            "skipped": False,
            "concordance": "UNCERTAIN",
            "observations": [],
            "support_level": "image_limited",
            "suspected_location": "Unspecified",
            "suspected_pattern": "Unspecified",
            "reasoning": (
                f"MedGemma vision timed out (>{SAFETY_AUDIT_TIMEOUT}s)."
            ),
            "latency_s": round(time.time() - start, 1),
            "error": f"Vision analysis timed out (>{SAFETY_AUDIT_TIMEOUT}s)",
        }
    except Exception as e:
        return {
            "skipped": False,
            "concordance": "UNCERTAIN",
            "observations": [],
            "support_level": "image_limited",
            "suspected_location": "Unspecified",
            "suspected_pattern": "Unspecified",
            "reasoning": f"Safety audit error: {str(e)}",
            "latency_s": round(time.time() - start, 1),
            "error": str(e),
        }


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
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
