"""Explainable recommendation helpers."""

from __future__ import annotations

from .models import Recommendation


def priority_from_score(score: float, high_threshold: float = 70.0) -> str:
    if score >= high_threshold:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def make_recommendation(
    label: str,
    reason: str,
    priority: str = "medium",
    confidence: float = 0.7,
    metric: str = "general",
) -> Recommendation:
    return Recommendation(
        label=label,
        reason=reason,
        priority=priority,
        confidence=round(max(0.0, min(1.0, confidence)), 2),
        metric=metric,
    )
