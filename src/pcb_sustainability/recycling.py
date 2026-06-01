"""End-of-life recovery estimation."""

from __future__ import annotations

from .models import ComponentScore, PCBFeatures, PCBScore, RecoveryEstimate
from .utils import clamp


def estimate_recovery(
    components: list[ComponentScore],
    pcb_features: PCBFeatures,
    pcb_score: PCBScore,
) -> RecoveryEstimate:
    """Estimate recoverable material categories and final recyclability score."""

    area = pcb_features.board_area_mm2 or 2500.0
    copper_area = pcb_features.copper_area_mm2 or area * 0.22 * max(pcb_features.layer_count or 2, 1)
    total_qty = sum(max(item.quantity, 1) for item in components) or max(pcb_features.component_count, 1)

    connector_like = sum(item.quantity for item in components if "connector" in item.description.lower() or "terminal" in item.description.lower())
    battery_like = sum(item.quantity for item in components if "battery" in item.description.lower() or "cell" in item.description.lower()) + pcb_features.battery_count
    ic_like = sum(item.quantity for item in components if any(term in item.description.lower() for term in ["ic", "microcontroller", "module", "processor"]))

    materials = {
        "copper_g": round(copper_area * 0.00896 * 0.035, 2),
        "gold_silver_tin_contacts_g": round((connector_like * 0.08) + (ic_like * 0.025), 2),
        "solder_g": round(total_qty * 0.045 + pcb_features.hole_count * 0.006, 2),
        "fr4_substrate_g": round(area * 0.00185, 2),
        "plastics_g": round(total_qty * 0.18 + connector_like * 0.45, 2),
        "aluminum_steel_g": round(connector_like * 0.25 + pcb_features.connector_count * 0.35, 2),
        "batteries_count": float(battery_like),
    }

    positive = []
    negative = []
    if pcb_features.connector_count or connector_like:
        positive.append("Connectors and modular parts improve component separation.")
    if pcb_features.edge_component_count:
        positive.append("Edge-placed components are easier to access during disassembly.")
    if pcb_score.material_recovery >= 65:
        positive.append("PCB material recovery estimate is acceptable for a prototype design.")
    if pcb_features.layer_count > 4:
        negative.append("High layer count lowers laminate and copper recovery efficiency.")
    if pcb_features.smd_ratio > 0.7:
        negative.append("High SMD ratio increases automated or thermal removal effort.")
    if battery_like:
        negative.append("Batteries require separate safe removal and recycling stream.")

    recovery_percent = clamp(
        48
        + pcb_score.material_recovery * 0.26
        + pcb_score.accessibility * 0.14
        - pcb_features.smd_ratio * 12
        - min(battery_like * 5, 15)
    )
    component_average = sum(item.score * max(item.quantity, 1) for item in components) / (total_qty or 1) if components else 60
    final_score = clamp(component_average * 0.48 + pcb_score.score * 0.32 + recovery_percent * 0.20)

    return RecoveryEstimate(
        final_score=round(final_score, 1),
        recovery_percent=round(recovery_percent, 1),
        materials=materials,
        positive_factors=positive or ["No major positive design-for-recovery factors were detected."],
        negative_factors=negative or ["No major end-of-life recovery blockers were detected."],
    )
