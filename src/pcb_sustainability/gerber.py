"""Prototype Gerber ZIP and placement analysis.

The parser intentionally uses conservative text heuristics instead of pretending to be a
full CAM engine. It extracts reliable demo features from RS-274X coordinate streams,
Excellon drill hits, layer filenames, and optional placement data.
"""

from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .models import PCBFeatures, PCBScore
from .recommendations import make_recommendation
from .utils import clamp


GERBER_EXTENSIONS = {".gbr", ".ger", ".gtl", ".gbl", ".gts", ".gbs", ".gto", ".gbo", ".gm1", ".gko", ".gml", ".pho"}
DRILL_EXTENSIONS = {".drl", ".xln", ".txt"}
COPPER_HINTS = ("gtl", "gbl", "g1", "g2", "inner", "in1", "in2", "copper")
OUTLINE_HINTS = ("gko", "gm1", "outline", "edge", "profile")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_coordinates(text: str) -> list[tuple[float, float]]:
    coords = []
    for match in re.finditer(r"X(-?\d+)Y(-?\d+)", text, re.IGNORECASE):
        x_raw, y_raw = match.groups()
        scale = 1000.0 if max(len(x_raw), len(y_raw)) <= 5 else 10000.0
        coords.append((int(x_raw) / scale, int(y_raw) / scale))
    return coords


def _bounding_box(coords: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    if not coords:
        return None
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return None
    return width, height, width * height


def _estimate_copper_area(text: str, board_area: float | None) -> float | None:
    if not board_area:
        return None
    draw_ops = len(re.findall(r"D0?1\*", text))
    flash_ops = len(re.findall(r"D0?3\*", text))
    density_factor = clamp((draw_ops * 0.0025 + flash_ops * 0.0035), 0.08, 0.72)
    return round(board_area * density_factor, 2)


def _count_drills(text: str) -> int:
    return len(re.findall(r"^X-?\d+Y-?\d+", text, flags=re.IGNORECASE | re.MULTILINE))


def _classify_layers(files: list[Path]) -> int:
    copper_layers = []
    for path in files:
        name = path.name.lower()
        if path.suffix.lower() in GERBER_EXTENSIONS and any(hint in name for hint in COPPER_HINTS):
            copper_layers.append(name)
    return len(set(copper_layers))


def _placement_features(placement: pd.DataFrame | None, features: PCBFeatures) -> None:
    if placement is None or placement.empty:
        return
    features.component_count = len(placement)
    package_text = " ".join(placement.get("package", pd.Series(dtype=str)).fillna("").astype(str)).lower()
    ref_text = " ".join(placement.get("reference", pd.Series(dtype=str)).fillna("").astype(str)).lower()

    smd_terms = ["smd", "qfn", "qfp", "bga", "sot", "0603", "0805", "0402", "0201", "soic"]
    th_terms = ["th", "tht", "through", "dip", "terminal", "connector"]
    features.smd_count = sum(1 for value in placement.get("package", []) if any(term in str(value).lower() for term in smd_terms))
    features.through_hole_count = sum(1 for value in placement.get("package", []) if any(term in str(value).lower() for term in th_terms))
    features.connector_count = sum(1 for value in placement.get("package", []) if "conn" in str(value).lower() or "terminal" in str(value).lower())
    features.battery_count = ref_text.count("bat") + package_text.count("battery")
    features.high_value_count = sum(ref_text.count(prefix) for prefix in ["u", "ic", "mcu", "mod"])

    if features.board_width_mm and features.board_height_mm and {"x_mm", "y_mm"}.issubset(placement.columns):
        edge_margin = min(features.board_width_mm, features.board_height_mm) * 0.12
        edge_rows = placement[
            (placement["x_mm"].fillna(features.board_width_mm / 2) <= edge_margin)
            | (placement["y_mm"].fillna(features.board_height_mm / 2) <= edge_margin)
            | (placement["x_mm"].fillna(0) >= features.board_width_mm - edge_margin)
            | (placement["y_mm"].fillna(0) >= features.board_height_mm - edge_margin)
        ]
        features.edge_component_count = len(edge_rows)


def parse_gerber_zip(source: str | Path | BinaryIO, placement: pd.DataFrame | None = None) -> PCBFeatures:
    """Parse a Gerber ZIP and return best-effort design-for-recycling features."""

    features = PCBFeatures()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(temp_path)

        files = [path for path in temp_path.rglob("*") if path.is_file()]
        features.parsed_files = [path.name for path in files]
        gerbers = [path for path in files if path.suffix.lower() in GERBER_EXTENSIONS]
        drills = [path for path in files if path.suffix.lower() in DRILL_EXTENSIONS and ("drl" in path.name.lower() or "drill" in path.name.lower())]
        features.layer_count = _classify_layers(gerbers)

        outline_candidates = [path for path in gerbers if any(hint in path.name.lower() for hint in OUTLINE_HINTS)]
        coordinate_sources = outline_candidates or gerbers
        all_coords: list[tuple[float, float]] = []
        for path in coordinate_sources:
            all_coords.extend(_extract_coordinates(_read_text(path)))
        bbox = _bounding_box(all_coords)
        if bbox:
            features.board_width_mm, features.board_height_mm, features.board_area_mm2 = [round(value, 2) for value in bbox]
        else:
            features.warnings.append("Could not infer board outline dimensions from Gerber coordinates.")

        copper_estimates = []
        for path in gerbers:
            if any(hint in path.name.lower() for hint in COPPER_HINTS):
                copper_estimates.append(_estimate_copper_area(_read_text(path), features.board_area_mm2))
        copper_values = [value for value in copper_estimates if value is not None]
        if copper_values:
            features.copper_area_mm2 = round(sum(copper_values), 2)

        for path in drills:
            count = _count_drills(_read_text(path))
            features.hole_count += count
        features.via_count = max(0, int(features.hole_count * 0.72))

        if not gerbers:
            features.warnings.append("No RS-274X Gerber files were recognized in the ZIP.")
        if not drills:
            features.warnings.append("No Excellon drill file was recognized; hole and via counts may be missing.")
        if features.layer_count == 0 and gerbers:
            features.layer_count = min(2, len(gerbers))
            features.warnings.append("Copper layer count was inferred from available Gerber files.")

    _placement_features(placement, features)
    return features


def merge_manual_features(features: PCBFeatures, manual: dict) -> PCBFeatures:
    """Fill missing parser values with user-provided UI values."""

    for key, value in manual.items():
        if value in (None, "") or not hasattr(features, key):
            continue
        current = getattr(features, key)
        if current in (None, 0, 0.0, ""):
            setattr(features, key, value)
    if features.board_area_mm2 in (None, 0) and features.board_width_mm and features.board_height_mm:
        features.board_area_mm2 = round(features.board_width_mm * features.board_height_mm, 2)
    return features


def score_pcb(features: PCBFeatures) -> PCBScore:
    """Score layout choices that affect disassembly and recycling."""

    recs = []
    layer_penalty = max(features.layer_count - 2, 0) * 4
    via_penalty = min(features.via_density * 5000, 18)
    smd_penalty = features.smd_ratio * 22
    density_penalty = min(features.component_density * 6000, 20)
    battery_penalty = min(features.battery_count * 10, 20)

    disassembly_difficulty = clamp(25 + layer_penalty + via_penalty + smd_penalty + density_penalty + battery_penalty)
    accessibility = clamp(70 + features.edge_component_count * 1.5 + features.connector_count * 2 - density_penalty - smd_penalty * 0.4)
    modularity = clamp(48 + features.connector_count * 5 + features.edge_component_count * 0.8 - max(features.layer_count - 2, 0) * 3)
    material_recovery = clamp(62 + (features.copper_area_mm2 or 0) / max(features.board_area_mm2 or 1, 1) * 10 - battery_penalty - layer_penalty)
    score = clamp((100 - disassembly_difficulty) * 0.32 + accessibility * 0.23 + modularity * 0.2 + material_recovery * 0.25)

    if features.layer_count > 4:
        recs.append(make_recommendation(
            "Low material recovery",
            "The board appears to use many copper layers, which improves electrical routing but makes laminate separation and material recovery harder.",
            "medium",
            0.68,
            "layout",
        ))
    if features.smd_ratio > 0.72:
        recs.append(make_recommendation(
            "Difficult to desolder",
            "The placement file suggests a high SMD ratio. Consider modular connectors or serviceable through-hole parts for high-failure components.",
            "medium",
            0.73,
            "disassembly",
        ))
    if features.via_density > 0.015:
        recs.append(make_recommendation(
            "High via density",
            "Dense via usage can indicate compact routing that is harder to rework and may reduce clean copper recovery.",
            "low",
            0.62,
            "layout",
        ))
    if features.battery_count:
        recs.append(make_recommendation(
            "Battery requires accessible removal",
            "Battery references were detected. Place batteries near an accessible edge and avoid permanent adhesive to improve safe end-of-life handling.",
            "high",
            0.78,
            "safety",
        ))
    if not recs:
        recs.append(make_recommendation(
            "PCB layout is recycling-friendly",
            "No major layer-count, density, battery, or accessibility warning was detected from the available design data.",
            "low",
            0.58,
            "summary",
        ))

    return PCBScore(
        score=round(score, 1),
        disassembly_difficulty=round(disassembly_difficulty, 1),
        material_recovery=round(material_recovery, 1),
        accessibility=round(accessibility, 1),
        modularity=round(modularity, 1),
        recommendations=recs,
    )
