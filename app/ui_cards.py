"""HTML/text card builders for triage, safety audit, and X-ray guard UI."""


def make_triage_card(result: dict) -> str:
    color_map = {
        "red": ("#dc2626", "#fff", "🔴"),
        "yellow": ("#d97706", "#000", "🟡"),
        "green": ("#16a34a", "#fff", "🟢"),
    }
    bg, fg, icon = color_map[result["triage_color"]]
    prob_pct = round(result["probability"] * 100, 1)

    return f"""
    <div style="padding:32px; border-radius:16px; background:{bg}; color:{fg};
                font-family:system-ui; text-align:center; margin:8px 0;">
        <div style="font-size:52px; margin-bottom:8px;">{icon}</div>
        <div style="font-size:24px; font-weight:700; letter-spacing:1px; margin-bottom:4px;">
            {result['triage_level']}
        </div>
        <div style="font-size:48px; font-weight:800; margin:8px 0;">
            {prob_pct}%
        </div>
        <div style="font-size:15px; opacity:0.9;">
            Fracture probability · {result['body_region']} X-ray
        </div>
    </div>
    """


def make_reasoning_text(result: dict) -> str:
    prob = result["probability"]
    region = result["body_region"]
    model = result["model_used"]
    latency = result["latency_ms"]

    lines = []
    lines.append(
        f"**Classification:** "
        f"{'Fracture detected' if prob >= 0.5 else 'No fracture detected'}"
    )
    lines.append(f"**Confidence:** {result['confidence']}")
    if result.get("region_auto_detected"):
        lines.append(f"**Body region:** {region} *(auto-detected by MedGemma)*")
    else:
        lines.append(f"**Body region:** {region}")
    lines.append(f"**Model:** {model}")
    lines.append(f"**Latency:** {latency}ms")
    lines.append("")

    if prob >= 0.70:
        lines.append(
            f"⚡ The model detected strong fracture-like patterns in this "
            f"{region.lower()} X-ray. This case should be **prioritized** "
            "for radiologist review."
        )
    elif prob >= 0.40:
        lines.append(
            f"The model detected some features consistent with fracture patterns "
            f"in the {region.lower()}, but with moderate uncertainty. Additional "
            "imaging or clinical correlation is recommended."
        )
    else:
        lines.append(
            f"The model did not detect strong fracture patterns in this "
            f"{region.lower()} X-ray. However, subtle or non-displaced fractures "
            "can be missed. Clinical judgment should always take precedence."
        )

    return "\n\n".join(lines)


def make_safety_audit_card(audit_result: dict) -> str:
    """Build HTML card for the safety audit cross-check result."""
    if audit_result.get("skipped"):
        reason = audit_result.get("reason", "Safety audit skipped")
        return f"""
        <div style="padding:16px; border-radius:12px; background:#374151; color:#9ca3af;
                    font-family:system-ui; margin:8px 0; border:1px solid #4b5563;">
            <div style="font-size:14px; font-weight:600; margin-bottom:4px;">
                ⏭️ Safety Audit Skipped
            </div>
            <div style="font-size:13px;">{reason}</div>
        </div>
        """

    concordance = audit_result.get("concordance", "UNCERTAIN")

    style_map = {
        "CONCORDANT": ("#065f46", "#d1fae5", "#047857", "✅", "Both models agree"),
        "DISCORDANT": (
            "#991b1b",
            "#fef2f2",
            "#dc2626",
            "⚠️",
            "Models DISAGREE — review needed",
        ),
        "UNCERTAIN": (
            "#92400e",
            "#fffbeb",
            "#d97706",
            "❓",
            "Cross-check inconclusive",
        ),
    }
    text_color, bg_color, border_color, icon, headline = style_map.get(
        concordance,
        style_map["UNCERTAIN"],
    )

    observations = audit_result.get("observations", [])[:5]
    obs_html = ""
    if observations:
        obs_items = "".join(f"<li>{obs}</li>" for obs in observations)
        obs_html = (
            '<div style="margin-top:8px;">'
            '<div style="font-size:12px; font-weight:600; margin-bottom:4px;">'
            "Visual Observations:</div>"
            f'<ul style="margin:0; padding-left:20px; font-size:12px;">'
            f"{obs_items}</ul></div>"
        )

    reasoning = audit_result.get("reasoning", "")
    reasoning_html = ""
    if reasoning:
        reasoning_html = (
            f'<div style="margin-top:8px; font-size:12px; '
            f'font-style:italic; opacity:0.85;">{reasoning}</div>'
        )

    error = audit_result.get("error")
    error_html = ""
    if error:
        error_html = (
            f'<div style="margin-top:6px; font-size:11px; '
            f'color:#dc2626;">Error: {error}</div>'
        )

    latency = audit_result.get("latency_s", "")
    latency_html = f" · {latency}s" if latency else ""

    border_style = (
        f"3px solid {border_color}"
        if concordance == "DISCORDANT"
        else f"1px solid {border_color}"
    )

    return f"""
    <div style="padding:16px; border-radius:12px; background:{bg_color}; color:{text_color};
                font-family:system-ui; margin:8px 0; border:{border_style};">
        <div style="font-size:16px; font-weight:700; margin-bottom:4px;">
            {icon} {headline}
        </div>
        <div style="font-size:13px; opacity:0.8;">
            MedGemma Visual Safety Audit{latency_html}
        </div>
        {obs_html}
        {reasoning_html}
        {error_html}
        <div style="margin-top:10px; font-size:11px; opacity:0.6;">
            This is an AI cross-check, not a diagnosis. Always defer to clinical judgment.
        </div>
    </div>
    """


def make_xray_guard_card(guard_result: dict) -> str:
    """Build HTML rejection card when the uploaded image is not an X-ray."""
    modality = guard_result.get("modality", "unknown").replace("_", " ").title()
    reason = guard_result.get("reason", "")
    latency = guard_result.get("latency_s", "")
    latency_html = f" · {latency}s" if latency else ""

    return f"""
    <div style="padding:24px; border-radius:14px; background:#1e293b; color:#cbd5e1;
                font-family:system-ui; text-align:center; margin:8px 0;
                border:2px solid #475569;">
        <div style="font-size:48px; margin-bottom:8px;">&#x1F6AB;</div>
        <div style="font-size:22px; font-weight:700; color:#f1f5f9; margin-bottom:6px;">
            Not an X-ray
        </div>
        <div style="font-size:15px; margin-bottom:12px; color:#94a3b8;">
            Detected modality: <strong style="color:#e2e8f0;">{modality}</strong>
        </div>
        <div style="font-size:13px; margin-bottom:16px; color:#94a3b8; font-style:italic;">
            {reason}
        </div>
        <div style="font-size:13px; color:#64748b;">
            Please upload a conventional plain-film X-ray (radiograph) for fracture analysis.
        </div>
        <div style="margin-top:10px; font-size:11px; color:#475569;">
            MedGemma X-ray Guard{latency_html}
        </div>
    </div>
    """
