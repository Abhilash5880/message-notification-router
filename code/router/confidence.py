"""Confidence calibration based on evidence and signal agreement."""

from __future__ import annotations

from typing import Iterable

from .retrieval import Evidence


def calibrate(
    action: str, *, strong_rule: bool, evidence: Iterable[Evidence],
    media_quality: str, ambiguous: bool, model_confidence: float | None = None,
) -> float:
    evidence = list(evidence)
    support = max((item.support for item in evidence), default=0.0)
    relevance = max((item.relevance for item in evidence), default=0.0)
    base = {"notify": 0.72, "digest": 0.70, "mute": 0.74}[action]
    if strong_rule:
        base += 0.15
    base += min(0.10, support * 0.10) + min(0.06, relevance * 0.04)
    if media_quality in {"metadata_only", "unavailable"}:
        base -= 0.06
    if ambiguous:
        base -= 0.05
    # Model confidence is advisory only: move the local value by at most 0.03.
    if model_confidence is not None and not strong_rule:
        bounded_delta = max(-0.12, min(0.12, float(model_confidence) - base))
        base += bounded_delta * 0.25
    return round(min(0.94, max(0.50, base)), 2)
