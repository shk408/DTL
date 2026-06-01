"""Shared dataclasses used across the sustainability pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Recommendation:
    """Human-readable recommendation with explainability metadata."""

    label: str
    reason: str
    priority: str = "medium"
    confidence: float = 0.7
    metric: str = "general"


@dataclass
class ComponentScore:
    """Sustainability result for one BoM line item."""

    part_number: str
    description: str
    quantity: int
    score: float
    toxicity_risk: float
    recyclability: float
    repairability: float
    restricted_substance_risk: float
    sourcing_availability: float
    obsolescence_risk: float
    greener_alternative_score: float
    recommendations: list[Recommendation] = field(default_factory=list)
    enrichment: dict[str, Any] = field(default_factory=dict)


@dataclass
class PCBFeatures:
    """Features extracted from PCB images, layout screenshots, placement files, or manual input."""

    board_width_mm: float | None = None
    board_height_mm: float | None = None
    board_area_mm2: float | None = None
    layer_count: int = 0
    copper_area_mm2: float | None = None
    hole_count: int = 0
    via_count: int = 0
    component_count: int = 0
    smd_count: int = 0
    through_hole_count: int = 0
    edge_component_count: int = 0
    connector_count: int = 0
    battery_count: int = 0
    high_value_count: int = 0
    warnings: list[str] = field(default_factory=list)
    parsed_files: list[str] = field(default_factory=list)

    @property
    def via_density(self) -> float:
        if not self.board_area_mm2:
            return 0.0
        return self.via_count / self.board_area_mm2

    @property
    def component_density(self) -> float:
        if not self.board_area_mm2:
            return 0.0
        return self.component_count / self.board_area_mm2

    @property
    def smd_ratio(self) -> float:
        total = self.smd_count + self.through_hole_count
        if total == 0:
            return 0.0
        return self.smd_count / total


@dataclass
class PCBScore:
    """Explainable design-for-recycling score for the physical PCB layout."""

    score: float
    disassembly_difficulty: float
    material_recovery: float
    accessibility: float
    modularity: float
    recommendations: list[Recommendation] = field(default_factory=list)


@dataclass
class RecoveryEstimate:
    """Rough end-of-life material recovery estimate."""

    final_score: float
    recovery_percent: float
    materials: dict[str, float]
    positive_factors: list[str]
    negative_factors: list[str]
