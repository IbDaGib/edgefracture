"""Safety-audit and region-detection helpers for MedGemma vision."""

import time

import requests
from PIL import Image


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

XRAY_GUARD_PROMPT = """\
You are a medical imaging assistant. Examine this image and determine \
whether it is a conventional X-ray (radiograph).

Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) \
using this exact schema:

{{"is_xray": true, "modality": "<detected imaging modality>", "reason": "<brief explanation>"}}

IMPORTANT:
- "is_xray" must be true ONLY for conventional plain-film X-rays (radiographs)
- Set "is_xray" to false for: photographs, CT scans, MRI scans, ultrasound, \
documents, diagrams, screenshots, non-medical images, or any other non-radiograph content
- "modality" should be one of: "xray", "ct", "mri", "ultrasound", "photograph", \
"document", "diagram", "unknown"
- Be conservative: if uncertain, set "is_xray" to false"""

COMMON_REGIONS = [
    "Hand",
    "Wrist",
    "Forearm",
    "Elbow",
    "Humerus",
    "Shoulder",
    "Clavicle",
    "Scapula",
    "Ribs",
    "Spine",
    "Pelvis",
    "Hip",
    "Femur",
    "Knee",
    "Leg",
    "Tibia",
    "Fibula",
    "Ankle",
    "Foot",
    "Skull",
    "Cervical",
    "Thoracic",
    "Lumbar",
    "Sacrum",
]


def check_vision_support(ollama_base_url: str, medgemma_model: str) -> bool:
    """Query Ollama to check if the configured MedGemma model is available."""
    try:
        resp = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if resp.ok:
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(medgemma_model in m or m in medgemma_model for m in models)
        return False
    except Exception:
        return False


def compute_concordance(
    cxr_probability: float,
    visual_assessment: str,
    triage_thresholds: dict,
) -> dict:
    """Compare CXR Foundation probability with MedGemma visual assessment."""
    if cxr_probability >= triage_thresholds["red"]:
        cxr_category = "fracture_likely"
    elif cxr_probability < triage_thresholds["yellow"]:
        cxr_category = "fracture_unlikely"
    else:
        cxr_category = "uncertain"

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


def detect_region_medgemma(
    image: Image.Image,
    *,
    is_vision_available,
    call_medgemma_vision,
    parse_json_response,
) -> tuple[str, bool]:
    """Detect body region from an X-ray image using MedGemma vision."""
    if not is_vision_available():
        return ("Unknown", False)

    try:
        data = call_medgemma_vision(REGION_DETECT_PROMPT, image)
        raw = data.get("response", "").strip()

        parsed = parse_json_response(raw)
        if parsed and parsed.get("region"):
            region = parsed["region"].strip().title()
            if region and region != "Unknown":
                return (region, True)

        if raw:
            raw_lower = raw.lower()
            for region in COMMON_REGIONS:
                if region.lower() in raw_lower:
                    return (region, True)

        return ("Unknown", False)
    except Exception:
        return ("Unknown", False)


def check_xray_image(
    image: Image.Image,
    *,
    xray_guard_enabled: bool,
    is_vision_available,
    call_medgemma_vision,
    parse_json_response,
) -> dict:
    """Check whether an image is a conventional X-ray using MedGemma vision.

    Returns dict with keys: is_xray, modality, reason, latency_s, skipped.
    Fails open (is_xray=True) when guard is disabled, vision unavailable,
    parse fails, or an exception occurs.
    """
    default_pass = {
        "is_xray": True,
        "modality": "unknown",
        "reason": "",
        "latency_s": 0.0,
        "skipped": True,
    }

    if not xray_guard_enabled:
        return {**default_pass, "reason": "X-ray guard disabled"}

    if not is_vision_available():
        return {**default_pass, "reason": "MedGemma vision not available"}

    start = time.time()
    try:
        data = call_medgemma_vision(XRAY_GUARD_PROMPT, image)
        raw = data.get("response", "").strip()
        parsed = parse_json_response(raw)

        latency = round(time.time() - start, 1)

        if parsed is None:
            return {
                "is_xray": True,
                "modality": "unknown",
                "reason": "Could not parse MedGemma response",
                "latency_s": latency,
                "skipped": False,
            }

        return {
            "is_xray": bool(parsed.get("is_xray", True)),
            "modality": str(parsed.get("modality", "unknown")).lower(),
            "reason": str(parsed.get("reason", "")),
            "latency_s": latency,
            "skipped": False,
        }
    except Exception:
        return {
            **default_pass,
            "latency_s": round(time.time() - start, 1),
            "reason": "Guard rail error — defaulting to pass",
            "skipped": False,
        }


def run_safety_audit(
    image: Image.Image,
    classification_result: dict,
    *,
    safety_audit_enabled: bool,
    is_vision_available,
    medgemma_model: str,
    call_medgemma_vision,
    parse_json_response,
    normalize_support_level,
    compute_concordance_fn,
    timeout_s: int,
) -> dict:
    """Run MedGemma vision analysis and compare with CXR Foundation result."""
    if not safety_audit_enabled:
        return {
            "skipped": True,
            "reason": "Safety audit disabled via SAFETY_AUDIT_ENABLED=false",
        }

    if not is_vision_available():
        return {
            "skipped": True,
            "reason": (
                f"MedGemma model not available. "
                f"Pull it with: ollama pull {medgemma_model}"
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
        data = call_medgemma_vision(prompt, image)
        raw_response = data.get("response", "").strip()

        parsed = parse_json_response(raw_response)
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
            support_level = normalize_support_level(parsed.get("support_level", ""))
            suspected_location = str(parsed.get("suspected_location", "")).strip() or "Unspecified"
            suspected_pattern = str(parsed.get("suspected_pattern", "")).strip() or "Unspecified"

        concordance_result = compute_concordance_fn(
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
            "reasoning": f"MedGemma vision timed out (>{timeout_s}s).",
            "latency_s": round(time.time() - start, 1),
            "error": f"Vision analysis timed out (>{timeout_s}s)",
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
