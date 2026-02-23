"""MedGemma reporting pipeline and prompt helpers."""

import json
import re
import time

import requests

try:
    from .config import CXR_SOFT_LOCK_MARGIN, TRIAGE_THRESHOLDS
except ImportError:
    from config import CXR_SOFT_LOCK_MARGIN, TRIAGE_THRESHOLDS


OVERCONFIDENT_PATTERNS = [
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

STRUCTURED_JSON_FIELDS_CLINICAL = [
    "anchor_impression",
    "anatomic_location",
    "likely_fracture_pattern",
    "urgency",
    "urgency_rationale",
    "recommended_imaging",
    "immediate_actions",
    "confidence_rationale",
    "discordance_note",
]
STRUCTURED_JSON_FIELDS_PATIENT = ["summary", "next_steps", "reassurance"]


def safety_gate(text: str) -> str:
    """Scan output for overconfident language and append safety disclaimer."""
    text = text.rstrip()
    for pattern in OVERCONFIDENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(
                pattern,
                lambda m: f"*{m.group(0)}*",
                text,
                flags=re.IGNORECASE,
            )
    if not text.endswith(SAFETY_DISCLAIMER.strip()):
        text += SAFETY_DISCLAIMER
    return text


def append_metadata_block(text: str, result: dict) -> str:
    """Programmatically append structured metadata."""
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


def sanitize_clinical_context(text: str) -> str:
    """Sanitize user-provided clinical context before prompt interpolation."""
    text = text.strip()[:500]

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

    text = re.sub(r"[{}<>|\\`]", "", text)
    return text.strip()


def build_context_section(clinical_context: str) -> str:
    if clinical_context and clinical_context.strip():
        sanitized = sanitize_clinical_context(clinical_context)
        if sanitized:
            return f"\nClinical context provided: {sanitized}\n"
    return ""


def build_audit_findings_section(audit_result: dict | None) -> str:
    """Format safety-audit findings into a prompt section."""
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


def build_clinical_summary_section(clinical_report: dict | None) -> str:
    """Format clinical report output into a patient-prompt section."""
    if not clinical_report or clinical_report.get("error"):
        return ""

    parsed = clinical_report.get("structured_json")
    if parsed:
        parsed = normalize_clinical_structured_json(parsed)
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

    narrative = clinical_report.get("report_text", "").strip()
    if narrative:
        truncated = narrative[:500]
        if len(narrative) > 500:
            truncated += "..."
        return (
            "\nClinical AI Summary (use as the basis for your explanation — "
            f"translate into patient-friendly language):\n{truncated}\n"
        )
    return ""


def parse_json_response(raw: str) -> dict | None:
    """Try to extract a JSON object from response text."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def normalize_support_level(value: str) -> str:
    """Normalize MedGemma support-level label to allowed enum."""
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


def resolve_cxr_anchor_decision(
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

    support_level = normalize_support_level((audit_result or {}).get("support_level", ""))
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
    discordant = concordance == "DISCORDANT" or support_level == "does_not_support_cxr"
    tempered_confidence = discordant or image_limited or concordance == "UNCERTAIN"

    near_threshold = any(
        abs(prob - threshold) <= threshold_margin for threshold in TRIAGE_THRESHOLDS.values()
    )

    downgrade_map = {"URGENT": "MODERATE", "MODERATE": "LOW", "LOW": "LOW"}
    urgency_adjusted = tempered_confidence and near_threshold and cxr_urgency != "LOW"
    final_urgency = downgrade_map[cxr_urgency] if urgency_adjusted else cxr_urgency

    if discordant:
        discordance_note = (
            "Visual audit is discordant with the CXR anchor; confidence is tempered."
        )
    elif image_limited:
        discordance_note = "Visual audit is image-limited/uncertain; confidence is tempered."
    elif concordance == "UNCERTAIN":
        discordance_note = (
            "Cross-check is uncertain; maintain CXR anchor with tempered confidence."
        )
    else:
        discordance_note = "Visual audit supports the CXR anchor."

    return {
        "cxr_probability_pct": round(prob * 100, 1),
        "classification_anchor": "Fracture detected" if prob >= 0.5 else "No fracture detected",
        "cxr_urgency": cxr_urgency,
        "final_urgency": final_urgency,
        "urgency_adjusted": urgency_adjusted,
        "tempered_confidence": tempered_confidence,
        "support_level": support_level,
        "concordance": concordance,
        "discordance_note": discordance_note,
    }


def build_cxr_anchor_section(
    classification_result: dict,
    anchor_context: dict | None,
    audit_result: dict | None = None,
) -> str:
    """Build compact CXR anchor context for clinical prompts."""
    if not anchor_context:
        return ""

    location = (audit_result or {}).get("suspected_location", "") or "Unspecified"
    pattern = (audit_result or {}).get("suspected_pattern", "") or "Unspecified"

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


def normalize_clinical_structured_json(
    parsed: dict,
    anchor_context: dict | None = None,
) -> dict:
    """Map new clinical schema fields to legacy keys for compatibility."""
    if not isinstance(parsed, dict):
        return {}

    normalized = dict(parsed)

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

    if not normalized.get("primary_finding") and normalized.get("anchor_impression"):
        normalized["primary_finding"] = normalized["anchor_impression"]
    if not normalized.get("differential") and normalized.get("likely_fracture_pattern"):
        normalized["differential"] = normalized["likely_fracture_pattern"]

    if not normalized.get("recommendation"):
        recommendation_parts = []
        if normalized.get("immediate_actions"):
            recommendation_parts.append(str(normalized["immediate_actions"]).strip())
        if normalized.get("recommended_imaging"):
            recommendation_parts.append(f"Imaging: {str(normalized['recommended_imaging']).strip()}")
        recommendation = " ".join(p for p in recommendation_parts if p).strip()
        if recommendation:
            normalized["recommendation"] = recommendation

    if not normalized.get("clinical_note"):
        note_parts = []
        if normalized.get("confidence_rationale"):
            note_parts.append(str(normalized["confidence_rationale"]).strip())
        discordance_note = normalized.get("discordance_note") or (anchor_context or {}).get(
            "discordance_note",
            "",
        )
        if discordance_note:
            note_parts.append(str(discordance_note).strip())
        clinical_note = " ".join(p for p in note_parts if p).strip()
        if clinical_note:
            normalized["clinical_note"] = clinical_note

    return normalized


def render_structured_clinical(parsed: dict) -> str:
    """Render parsed clinical JSON into readable markdown."""
    parsed = normalize_clinical_structured_json(parsed)

    lines = []
    lines.append("**CXR-Anchored Clinical Synthesis**")

    if parsed.get("anchor_impression") or parsed.get("primary_finding"):
        lines.append(f"**Impression:** {parsed.get('anchor_impression') or parsed.get('primary_finding')}")
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

    if parsed.get("recommendation") and not parsed.get("immediate_actions"):
        lines.append(f"**Recommendation:** {parsed['recommendation']}")
    if parsed.get("clinical_note") and not parsed.get("confidence_rationale"):
        lines.append(f"**Note:** {parsed['clinical_note']}")

    rendered = "\n\n".join(line for line in lines if line)
    return rendered if rendered.strip() != "**CXR-Anchored Clinical Synthesis**" else ""


def render_structured_patient(parsed: dict) -> str:
    """Render parsed patient JSON into friendly markdown."""
    lines = []
    if parsed.get("summary"):
        lines.append(parsed["summary"])
    if parsed.get("next_steps"):
        lines.append(f"**What happens next:** {parsed['next_steps']}")
    if parsed.get("reassurance"):
        lines.append(f"*{parsed['reassurance']}*")
    return "\n\n".join(lines)


def clean_medgemma_output(text: str) -> str:
    """Strip MedGemma artifacts: leaked prompts, critique loops, unused tokens."""
    text = re.sub(r"<unused\d+>", "", text)
    if "You are a musculoskeletal" in text or "You are a friendly" in text:
        for marker in [
            "1.",
            "CLINICAL SUMMARY",
            "Clinical Summary",
            "The AI",
            "Your X-ray",
            "Based on",
        ]:
            idx = text.find(marker)
            if idx > 0:
                text = text[idx:]
                break
    text = re.sub(r"\*\*Critique:?\*\*.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*Self-assessment:?\*\*.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _synthesize_fallback_json(
    report_type: str,
    classification_result: dict,
    anchor_context: dict | None,
) -> dict:
    """Build structured JSON from programmatic data when MedGemma fails to produce JSON."""
    prob = classification_result["probability"]
    body_region = classification_result.get("body_region", "Unknown")
    classification = "Fracture detected" if prob >= 0.5 else "No fracture detected"
    confidence = classification_result.get("confidence", "Unknown")
    triage_level = classification_result.get("triage_level", "Unknown")

    if report_type == "clinical":
        fallback: dict = {
            "anchor_impression": classification,
            "anatomic_location": body_region,
            "urgency": (anchor_context or {}).get("final_urgency", triage_level.upper()),
            "confidence_rationale": f"CXR model confidence: {confidence} ({round(prob * 100, 1)}%)",
            "discordance_note": (anchor_context or {}).get("discordance_note", ""),
        }
        if anchor_context:
            if anchor_context.get("urgency_adjusted"):
                fallback["urgency_rationale"] = (
                    "Urgency was adjusted based on cross-check with visual audit."
                )
            fallback["likely_fracture_pattern"] = (
                "Requires further imaging for definitive characterisation."
            )
        if prob >= 0.5:
            fallback["immediate_actions"] = (
                "Correlate clinically; consider additional imaging as indicated."
            )
        else:
            fallback["immediate_actions"] = (
                "Routine follow-up; re-image if symptoms persist."
            )
        return fallback

    # Patient report
    if prob >= 0.5:
        summary = (
            f"The X-ray analysis of your {body_region.lower()} suggests a possible fracture "
            f"(confidence: {confidence.lower()})."
        )
        next_steps = (
            "Your doctor will review these results and may order additional imaging "
            "to get a clearer picture."
        )
        reassurance = (
            "Finding a possible fracture early means your care team can act quickly. "
            "You are in good hands."
        )
    else:
        summary = (
            f"The X-ray analysis of your {body_region.lower()} did not detect signs of a fracture "
            f"(confidence: {confidence.lower()})."
        )
        next_steps = (
            "Your doctor will review these results. If your symptoms continue, "
            "they may recommend further evaluation."
        )
        reassurance = (
            "A negative result is reassuring, but always follow up with your doctor "
            "if you have ongoing pain or concerns."
        )

    return {"summary": summary, "next_steps": next_steps, "reassurance": reassurance}


def generate_report(
    classification_result: dict,
    clinical_context: str = "",
    report_type: str = "clinical",
    audit_result: dict | None = None,
    clinical_summary: dict | None = None,
    *,
    call_medgemma,
) -> dict:
    """Generate a structured + narrative report via MedGemma."""
    prob = classification_result["probability"]
    anchor_context = None
    if report_type == "clinical":
        anchor_context = resolve_cxr_anchor_decision(classification_result, audit_result)

    anchor_context_section = (
        build_cxr_anchor_section(classification_result, anchor_context, audit_result)
        if report_type == "clinical"
        else ""
    )
    context_section = build_context_section(clinical_context)
    audit_section = (
        build_audit_findings_section(audit_result) if report_type == "clinical" else ""
    )
    clinical_section = (
        build_clinical_summary_section(clinical_summary) if report_type == "patient" else ""
    )

    prompt_kwargs = dict(
        body_region=classification_result["body_region"],
        score_pct=round(prob * 100, 1),
        triage_level=classification_result["triage_level"],
        classification="Fracture detected" if prob >= 0.5 else "No fracture detected",
        confidence=classification_result["confidence"],
        anchor_context_section=anchor_context_section,
        clinical_context_section=context_section,
        audit_findings_section=audit_section,
        clinical_summary_section=clinical_section,
    )

    json_template = CLINICAL_JSON_PROMPT if report_type == "clinical" else PATIENT_JSON_PROMPT
    narrative_template = CLINICAL_PROMPT if report_type == "clinical" else PATIENT_PROMPT
    render_fn = render_structured_clinical if report_type == "clinical" else render_structured_patient

    start = time.time()
    try:
        json_prompt = json_template.format(**prompt_kwargs)
        data = call_medgemma(json_prompt)
        raw_response = data.get("response", "").strip()
        parsed = parse_json_response(raw_response)

        if parsed is not None:
            if report_type == "clinical":
                parsed = normalize_clinical_structured_json(parsed, anchor_context=anchor_context)
            report_text = render_fn(parsed)
        else:
            # Synthesize structured JSON from programmatic data
            parsed = _synthesize_fallback_json(
                report_type, classification_result, anchor_context
            )
            narrative_prompt = narrative_template.format(**prompt_kwargs)
            data = call_medgemma(narrative_prompt)
            report_text = clean_medgemma_output(data.get("response", "").strip())

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
