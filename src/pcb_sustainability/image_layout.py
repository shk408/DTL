
"""Lightweight PCB image/layout analysis for coursework demos.

This module powers the fast-mode version. It uses PIL and
simple image statistics to estimate board dimensions, component density, and
layout complexity from PCB photos or layout screenshots.
"""

from __future__ import annotations

from collections import deque
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

from .models import PCBFeatures, PCBScore
from .recommendations import make_recommendation
from .utils import clamp


def _open_image(source: str | Path | BinaryIO) -> tuple[Image.Image, str]:
    if hasattr(source, "read"):
        raw = source.read()
        name = getattr(source, "name", "uploaded_image")
        image = Image.open(BytesIO(raw))
        return image, name
    path = Path(source)
    image = Image.open(path)
    return image, path.name


def _prepare_array(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    max_dim = 320
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))
    gray = image.convert("L")
    return np.asarray(gray, dtype=np.uint8)


def _component_stats(mask: np.ndarray, min_area: int = 18) -> list[dict[str, float]]:
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, float]] = []
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))

    starts = np.argwhere(mask & ~visited)
    for start_row, start_col in starts:
        if visited[start_row, start_col] or not mask[start_row, start_col]:
            continue
        q = deque([(int(start_row), int(start_col))])
        visited[start_row, start_col] = True

        area = 0
        min_r = max_r = int(start_row)
        min_c = max_c = int(start_col)
        touches_edge = False

        while q:
            r, c = q.popleft()
            area += 1
            if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                touches_edge = True
            if r < min_r:
                min_r = r
            if r > max_r:
                max_r = r
            if c < min_c:
                min_c = c
            if c > max_c:
                max_c = c
            for dr, dc in offsets:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        if area >= min_area:
            width = max(1, max_c - min_c + 1)
            height = max(1, max_r - min_r + 1)
            components.append(
                {
                    "area": float(area),
                    "width": float(width),
                    "height": float(height),
                    "aspect": float(max(width / height, height / width)),
                    "touches_edge": float(touches_edge),
                }
            )
    return components


def _pick_foreground_mask(gray: np.ndarray) -> tuple[np.ndarray, str, float]:
    low_threshold = float(np.percentile(gray, 35))
    high_threshold = float(np.percentile(gray, 65))
    candidates = [
        (gray < low_threshold, "dark", low_threshold),
        (gray > high_threshold, "light", high_threshold),
    ]
    scored = []
    for mask, label, threshold in candidates:
        components = _component_stats(mask)
        meaningful = [comp for comp in components if 18 <= comp["area"] <= 12000]
        score = len(meaningful) + sum(1 for comp in meaningful if comp["touches_edge"]) * 0.1
        foreground_fraction = float(mask.mean())
        if foreground_fraction < 0.02 or foreground_fraction > 0.8:
            score *= 0.75
        scored.append((score, mask, label, threshold))
    best = max(scored, key=lambda item: item[0])
    return best[1], best[2], best[3]


def parse_pcb_image(source: str | Path | BinaryIO) -> PCBFeatures:
    """Extract best-effort PCB features from a board image or layout screenshot."""

    image, name = _open_image(source)
    gray = _prepare_array(image)
    mask, mask_type, threshold = _pick_foreground_mask(gray)
    components = _component_stats(mask)

    width_px, height_px = image.size
    scale = 20.0
    board_width_mm = round(width_px / scale, 2)
    board_height_mm = round(height_px / scale, 2)
    board_area_mm2 = round(board_width_mm * board_height_mm, 2)

    foreground_fraction = float(mask.mean())
    component_count = len(components)
    smd_count = sum(1 for comp in components if comp["area"] < 120 or max(comp["width"], comp["height"]) < 18)
    through_hole_count = sum(
        1 for comp in components
        if 120 <= comp["area"] < 900 and comp["aspect"] < 3.2
    )
    connector_count = sum(
        1 for comp in components
        if comp["area"] >= 700 and comp["aspect"] >= 2.0
    )
    edge_component_count = sum(1 for comp in components if comp["touches_edge"])
    high_value_count = sum(1 for comp in components if comp["area"] >= 1000)
    hole_count = max(8, int(round(component_count * 1.1 + connector_count * 1.8)))
    via_count = max(0, int(round(hole_count * 0.65)))
    layer_count = 2 + min(4, max(0, int(round(component_count / 16))))
    copper_area_mm2 = round(board_area_mm2 * clamp(0.18 + foreground_fraction * 0.38, 0.12, 0.78), 2)

    warnings = [
        "Image-based analysis uses heuristic feature extraction. Add a placement file or manual inputs for exact counts.",
        f"Foreground detection used the {mask_type} mask at threshold {threshold:.1f}.",
    ]
    if component_count == 0:
        warnings.append("No clear component blobs were detected; the image may be too small, blurred, or a screenshot with low contrast.")

    return PCBFeatures(
        board_width_mm=board_width_mm,
        board_height_mm=board_height_mm,
        board_area_mm2=board_area_mm2,
        layer_count=layer_count,
        copper_area_mm2=copper_area_mm2,
        hole_count=hole_count,
        via_count=via_count,
        component_count=component_count,
        smd_count=smd_count,
        through_hole_count=through_hole_count,
        edge_component_count=edge_component_count,
        connector_count=connector_count,
        battery_count=0,
        high_value_count=high_value_count,
        warnings=warnings,
        parsed_files=[name],
    )


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
            "The image suggests a high SMD ratio. Consider modular connectors or serviceable through-hole parts for high-failure components.",
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
            "No major layer-count, density, battery, or accessibility warning was detected from the available design image.",
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
