"""Rules-based BoM sustainability scoring with an ML-ready interface."""

from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

import pandas as pd

from .models import ComponentScore, Recommendation
from .recommendations import make_recommendation
from .utils import clamp, normalize_text


TOXIC_KEYWORDS = {
    "lead": 35,
    "pb": 25,
    "cadmium": 45,
    "mercury": 50,
    "brominated": 30,
    "pvc": 20,
    "lithium": 25,
    "battery": 30,
    "relay": 12,
}

RECYCLABLE_KEYWORDS = {
    "connector": 15,
    "screw": 12,
    "terminal": 12,
    "aluminum": 15,
    "steel": 12,
    "copper": 16,
    "through": 10,
    "module": 12,
}

HARD_TO_REPAIR_KEYWORDS = {
    "bga": 30,
    "qfn": 20,
    "0402": 18,
    "0201": 25,
    "underfill": 35,
    "glue": 30,
    "potted": 45,
    "shield": 12,
}

GREENER_HINTS = {
    "lead free": 18,
    "lead-free": 18,
    "rohs": 16,
    "halogen free": 15,
    "halogen-free": 15,
    "reach": 10,
}

OBSOLESCENCE_TERMS = {
    "obsolete": 50,
    "nrnd": 35,
    "not recommended": 35,
    "eol": 45,
    "discontinued": 45,
}

PACKAGE_REPAIRABILITY = {
    "through": 88,
    "tht": 86,
    "dip": 85,
    "sop": 66,
    "soic": 68,
    "sot": 62,
    "qfp": 56,
    "qfn": 42,
    "bga": 24,
    "0201": 30,
    "0402": 42,
    "0603": 56,
    "0805": 66,
    "1206": 72,
}


class SustainabilityModel(Protocol):
    """Future ML models can implement this protocol."""

    def score_component(self, row: pd.Series, enrichment: dict) -> ComponentScore:
        ...


def _contains_score(text: str, weights: dict[str, int]) -> float:
    text_lower = text.lower()
    return sum(weight for keyword, weight in weights.items() if keyword in text_lower)


def _best_package_score(text: str) -> float:
    text_lower = text.lower()
    for key, score in PACKAGE_REPAIRABILITY.items():
        if key in text_lower:
            return score
    return 58.0


def _availability_score(enrichment: dict) -> float:
    if not enrichment:
        return 52.0
    status = normalize_text(enrichment.get("availability") or enrichment.get("stock_status")).lower()
    if any(term in status for term in ["in stock", "available", "add to cart"]):
        return 86.0
    if any(term in status for term in ["out of stock", "unavailable", "sold out"]):
        return 24.0
    if enrichment.get("price"):
        return 68.0
    return 50.0


class RuleBasedSustainabilityScorer:
    """Deterministic scoring engine for component-level sustainability."""

    weights = {
        "toxicity": 0.20,
        "recyclability": 0.18,
        "repairability": 0.18,
        "restricted": 0.16,
        "availability": 0.12,
        "obsolescence": 0.10,
        "greener": 0.06,
    }

    def score_component(self, row: pd.Series, enrichment: dict | None = None) -> ComponentScore:
        enrichment = enrichment or {}
        searchable = " ".join(
            normalize_text(row.get(col, ""))
            for col in ["part_number", "value", "footprint", "manufacturer", "description", "material", "compliance", "lifecycle"]
        )
        searchable = f"{searchable} {enrichment.get('title', '')} {enrichment.get('category', '')}"

        toxic_raw = _contains_score(searchable, TOXIC_KEYWORDS)
        green_raw = _contains_score(searchable, GREENER_HINTS)
        hard_raw = _contains_score(searchable, HARD_TO_REPAIR_KEYWORDS)
        obsolete_raw = _contains_score(searchable, OBSOLESCENCE_TERMS)

        toxicity_risk = clamp(toxic_raw - green_raw * 0.35)
        restricted_risk = clamp(toxic_raw * 0.85 - green_raw * 0.55)
        recyclability = clamp(55 + _contains_score(searchable, RECYCLABLE_KEYWORDS) - toxicity_risk * 0.35)
        repairability = clamp(_best_package_score(searchable) - hard_raw * 0.45)
        availability = _availability_score(enrichment)
        obsolescence_risk = clamp(obsolete_raw + (18 if availability < 35 else 0))
        greener_alternative_score = clamp(55 + green_raw - toxicity_risk * 0.2)

        score = (
            (100 - toxicity_risk) * self.weights["toxicity"]
            + recyclability * self.weights["recyclability"]
            + repairability * self.weights["repairability"]
            + (100 - restricted_risk) * self.weights["restricted"]
            + availability * self.weights["availability"]
            + (100 - obsolescence_risk) * self.weights["obsolescence"]
            + greener_alternative_score * self.weights["greener"]
        )

        recommendations = self._recommend(row, enrichment, {
            "toxicity_risk": toxicity_risk,
            "restricted_risk": restricted_risk,
            "recyclability": recyclability,
            "repairability": repairability,
            "availability": availability,
            "obsolescence_risk": obsolescence_risk,
            "greener_alternative_score": greener_alternative_score,
        })

        return ComponentScore(
            part_number=normalize_text(row.get("part_number")) or normalize_text(row.get("value")) or "Unknown",
            description=normalize_text(row.get("description")) or normalize_text(enrichment.get("title")) or "No description",
            quantity=int(row.get("quantity", 1) or 1),
            score=round(clamp(score), 1),
            toxicity_risk=round(toxicity_risk, 1),
            recyclability=round(recyclability, 1),
            repairability=round(repairability, 1),
            restricted_substance_risk=round(restricted_risk, 1),
            sourcing_availability=round(availability, 1),
            obsolescence_risk=round(obsolescence_risk, 1),
            greener_alternative_score=round(greener_alternative_score, 1),
            recommendations=recommendations,
            enrichment=enrichment,
        )

    def _recommend(self, row: pd.Series, enrichment: dict, metrics: dict[str, float]) -> list[Recommendation]:
        recs: list[Recommendation] = []
        text = " ".join(str(row.get(col, "")) for col in row.index).lower()

        if metrics["toxicity_risk"] >= 35:
            recs.append(make_recommendation(
                "High toxicity risk",
                "The item mentions substances or component types commonly associated with hazardous handling, such as lead, cadmium, mercury, brominated materials, PVC, or batteries.",
                "high",
                0.82,
                "toxicity",
            ))
        if metrics["restricted_risk"] >= 30 and "rohs" not in text:
            recs.append(make_recommendation(
                "Use RoHS-friendly option",
                "Restricted substance risk is elevated and the BoM line does not clearly state RoHS, lead-free, or halogen-free compliance.",
                "high",
                0.78,
                "restricted_substances",
            ))
        if metrics["repairability"] < 50:
            recs.append(make_recommendation(
                "Difficult to desolder",
                "The package or description suggests small SMD, BGA/QFN, glue, potting, shielding, or underfill that makes replacement harder.",
                "medium",
                0.74,
                "repairability",
            ))
        if metrics["recyclability"] < 50:
            recs.append(make_recommendation(
                "Better recyclable alternative available",
                "The component appears hard to separate or contains materials that reduce recovery value. Prefer modular connectors, through-hole service parts, or clearly marked recyclable materials.",
                "medium",
                0.67,
                "recyclability",
            ))
        if metrics["availability"] < 40:
            recs.append(make_recommendation(
                "Availability on Robu.in is poor",
                "Marketplace enrichment indicates the part may be out of stock or unavailable, increasing replacement and maintenance risk.",
                "medium",
                0.72,
                "availability",
            ))
        if metrics["obsolescence_risk"] >= 35:
            recs.append(make_recommendation(
                "High obsolescence risk",
                "The lifecycle/status text suggests obsolete, discontinued, EOL, or NRND supply conditions.",
                "high",
                0.8,
                "obsolescence",
            ))
        if not recs:
            recs.append(make_recommendation(
                "Acceptable sustainability profile",
                "No major toxicity, restricted substance, repairability, availability, or obsolescence warning was detected from the available fields.",
                "low",
                0.62,
                "summary",
            ))
        return recs


def score_bom(df: pd.DataFrame, enrichments: dict[str, dict] | None = None) -> tuple[list[ComponentScore], dict]:
    """Score all BoM rows and return component results plus summary metrics."""

    scorer = RuleBasedSustainabilityScorer()
    enrichments = enrichments or {}
    results = []
    weighted_scores = []
    for _, row in df.iterrows():
        key = normalize_text(row.get("part_number")) or normalize_text(row.get("description"))
        result = scorer.score_component(row, enrichments.get(key, {}))
        results.append(result)
        weighted_scores.append(result.score * max(result.quantity, 1))

    total_qty = sum(max(item.quantity, 1) for item in results) or 1
    summary = {
        "component_count": len(results),
        "total_quantity": total_qty,
        "summary_score": round(sum(weighted_scores) / total_qty, 1) if results else 0.0,
        "high_priority_recommendations": sum(
            1 for item in results for rec in item.recommendations if rec.priority == "high"
        ),
    }
    return results, summary


def component_scores_to_dataframe(scores: list[ComponentScore]) -> pd.DataFrame:
    """Flatten component scores for display and CSV export."""

    rows = []
    for score in scores:
        data = asdict(score)
        data["recommendations"] = "; ".join(f"{rec.label}: {rec.reason}" for rec in score.recommendations)
        data["robu_title"] = score.enrichment.get("title", "")
        data["robu_availability"] = score.enrichment.get("availability", "")
        data["robu_price"] = score.enrichment.get("price", "")
        data.pop("enrichment", None)
        rows.append(data)
    return pd.DataFrame(rows)
