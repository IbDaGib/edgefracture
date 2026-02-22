"""Prediction audit logging helpers."""

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


class PredictionLogger:
    """JSON-lines audit logger for all predictions.

    Writes one JSON object per line to a rotating log file. Privacy-safe:
    never logs images or full clinical-context text.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("edgefracture.predictions")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = RotatingFileHandler(
                self.log_dir / "predictions.jsonl",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def log_prediction(
        self,
        result: dict,
        body_region_input: str = "Unknown",
        clinical_context: str = "",
        report_type: str = "",
    ):
        """Record a single prediction to the audit log."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "probability": result.get("probability"),
            "triage_level": result.get("triage_level"),
            "body_region": result.get("body_region"),
            "body_region_input": body_region_input,
            "region_auto_detected": result.get("region_auto_detected"),
            "model_used": result.get("model_used"),
            "latency_ms": result.get("latency_ms"),
            "confidence": result.get("confidence"),
            "image_warnings": result.get("image_warnings"),
        }
        if report_type:
            entry["report_type"] = report_type
        if clinical_context:
            entry["had_clinical_context"] = True
        self.logger.info(json.dumps(entry))


prediction_logger = PredictionLogger()
