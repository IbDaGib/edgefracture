"""Evidence and transparency helpers for the app UI."""

try:
    from .config import VALIDATED_REGIONS
except ImportError:
    from config import VALIDATED_REGIONS


def get_model_transparency_info(result: dict) -> str:
    """Build a transparency explanation of how the AI reached its conclusion."""
    prob = result["probability"]
    region = result["body_region"]

    region_context = {
        "Hand": "metacarpal fractures, phalangeal fractures, boxer's fractures, and scaphoid fractures",
        "Leg": "tibial shaft fractures, fibula fractures, ankle fractures, and stress fractures",
        "Hip": "femoral neck fractures, intertrochanteric fractures, and acetabular fractures",
        "Shoulder": "proximal humerus fractures, clavicle fractures, and scapular fractures",
    }

    lines = []
    lines.append("### How This Result Was Produced\n")
    lines.append(
        "**Step 0 — Region Detection:** "
        "[MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) "
        "analyzed the X-ray image to automatically identify the body region "
        f"(**{region}**). This uses MedGemma's vision capability to classify "
        "the anatomical area before fracture screening begins.\n"
    )
    lines.append(
        "**Step 1 — Image Analysis:** Your X-ray was processed by "
        "[CXR Foundation](https://huggingface.co/google/cxr-foundation), "
        "a medical imaging model from Google Health AI. This model was originally "
        "trained on **821,000 chest X-rays** and produces rich visual embeddings "
        "that capture bone and tissue structure.\n"
    )
    lines.append(
        "**Step 2 — Fracture Screening:** The image embeddings (88,064 dimensions) "
        "were passed through a logistic regression classifier trained on the "
        "[FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012) dataset "
        "(4,083 musculoskeletal X-rays across hand, leg, hip, and shoulder). "
        f"The classifier estimated a **{round(prob * 100, 1)}% probability** of fracture.\n"
    )
    lines.append(
        "**Step 3 — Clinical Report:** "
        "[MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it), "
        "a medical language model, interpreted the screening score and generated "
        "a clinical summary. MedGemma also performs a visual safety audit, "
        "independently assessing the X-ray image to cross-check the CXR Foundation result.\n"
    )
    lines.append("### What This Means\n")

    if prob >= 0.70:
        lines.append(
            f"A score of **{round(prob * 100, 1)}%** indicates high suspicion of fracture. "
            f"Common fractures in the {region.lower()} include "
            f"{region_context.get(region, f'various fracture patterns typical of the {region.lower()}')}. "
            "**This case should be prioritized for radiologist review.**"
        )
    elif prob >= 0.40:
        lines.append(
            f"A score of **{round(prob * 100, 1)}%** falls in the uncertain range. "
            f"The model detected features that partially match fracture patterns in the {region.lower()} "
            f"(common types: {region_context.get(region, f'various fracture patterns typical of the {region.lower()}')}). "
            "Additional imaging or clinical correlation is recommended."
        )
    else:
        lines.append(
            f"A score of **{round(prob * 100, 1)}%** suggests low likelihood of fracture. "
            f"However, some fracture types in the {region.lower()} "
            "(e.g., hairline or non-displaced fractures) can be subtle. "
            "Clinical correlation is always advised."
        )

    lines.append("\n### Important Limitations\n")
    lines.append(
        "- CXR Foundation was trained on **chest X-rays only** — its application to "
        "musculoskeletal imaging represents a novel transfer learning approach\n"
        "- This is a **screening tool**, not a diagnostic device\n"
        "- Performance varies by body region and fracture type\n"
        "- Always requires review by a qualified healthcare professional"
    )

    if region not in VALIDATED_REGIONS:
        lines.append(
            f"\n- **Unvalidated region:** The linear probe was trained on Hand, Leg, Hip, "
            f"and Shoulder X-rays only (FracAtlas). Performance on **{region}** X-rays "
            f"has not been formally validated — interpret with additional caution"
        )

    return "\n".join(lines)


def get_performance_context() -> str:
    """Model performance data from actual experiments."""
    lines = []
    lines.append("### Model Performance\n")
    lines.append(
        "CXR Foundation was trained exclusively on chest X-rays, yet demonstrates "
        "strong transfer to musculoskeletal fracture detection — a task it was "
        "**never designed for**.\n"
    )
    lines.append("#### Per-Region AUC (Linear Probe, 5-fold CV)\n")
    lines.append("| Body Region | Images | Fractures | AUC | 95% Bootstrap CI |")
    lines.append("|-------------|--------|-----------|-----|------------------|")
    lines.append("| Hand | 1,510 | 438 | **0.850** | [0.823, 0.876] |")
    lines.append("| Hip | 179 | 10 | **0.864** | [0.706, 0.972] * |")
    lines.append("| Leg | 2,237 | 259 | **0.888** | [0.861, 0.913] |")
    lines.append("| Shoulder | 98 | 10 | **0.848** | [0.667, 0.976] * |")
    lines.append("| **Overall** | **4,024** | **717** | **0.882** | [0.864, 0.899] |")
    lines.append("")
    lines.append(
        "*\\* Hip and Shoulder each have only 10 fracture cases, producing wide "
        "confidence intervals (CI width > 0.15). Their AUC point estimates should "
        "be interpreted with caution — the true AUC could plausibly range from "
        "~0.67 to ~0.98. More labeled data for these regions would substantially "
        "narrow the uncertainty.*"
    )

    lines.append("\n#### Data Efficiency — AUC vs. Training Examples\n")
    lines.append(
        "How many labeled X-rays does it take to make a chest X-ray model "
        "useful for fracture screening?\n"
    )
    lines.append("| Training Examples | AUC | Sensitivity | Specificity |")
    lines.append("|-------------------|-----|-------------|-------------|")
    lines.append("| 10 | 0.556 | 0.004 | 0.999 |")
    lines.append("| 25 | 0.578 | 0.015 | 0.991 |")
    lines.append("| 50 | 0.607 | 0.078 | 0.964 |")
    lines.append("| 100 | 0.683 | 0.191 | 0.947 |")
    lines.append("| 250 | 0.785 | 0.421 | 0.912 |")
    lines.append("| 500 | 0.820 | 0.561 | 0.893 |")
    lines.append("| **4,024 (all)** | **0.882** | **0.692** | **0.917** |")
    lines.append(
        "\n*With just 500 labeled examples, the system crosses the 0.80 AUC threshold "
        "— demonstrating that pre-trained chest X-ray embeddings encode transferable "
        "representations for musculoskeletal fracture detection.*"
    )

    lines.append(
        "\n*Performance data covers the 4 body regions in the FracAtlas dataset. "
        "For X-rays from other body regions, the model has not been formally evaluated. "
        "The CXR Foundation embeddings may still transfer effectively, but AUC estimates "
        "are unavailable.*"
    )

    lines.append("\n### Edge Deployment\n")
    lines.append("| Metric | Target | Status |")
    lines.append("|--------|--------|--------|")
    lines.append("| Device | Jetson Orin Nano 8GB | ✓ |")
    lines.append("| Device cost | $249 | ✓ |")
    lines.append("| CXR Foundation latency | < 3s | *pending Jetson benchmarks* |")
    lines.append("| MedGemma report latency | < 90s | *pending Jetson benchmarks* |")
    lines.append("| Total memory (both models) | < 6GB | *pending Jetson benchmarks* |")
    lines.append("| Internet required | None | ✓ Fully offline |")

    return "\n".join(lines)
