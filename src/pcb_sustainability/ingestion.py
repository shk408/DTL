"""File ingestion for BoM and optional placement data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd


CANONICAL_COLUMNS = {
    "part_number": ["part number", "part_number", "mpn", "manufacturer part number", "part", "sku"],
    "value": ["value", "rating", "capacitance", "resistance"],
    "footprint": ["footprint", "package", "case", "land pattern"],
    "manufacturer": ["manufacturer", "mfr", "brand"],
    "description": ["description", "desc", "item", "name"],
    "quantity": ["quantity", "qty", "count"],
    "material": ["material", "materials"],
    "compliance": ["compliance", "rohs", "reach", "halogen free", "lead free"],
    "lifecycle": ["lifecycle", "status", "obsolete"],
    "reference": ["reference", "designator", "refdes", "refs"],
    "supplier_url": ["supplier url", "supplier_url", "supplier link", "robu url", "product url", "url", "link"],
}


def _read_table(source: str | Path | BinaryIO, suffix: str | None = None) -> pd.DataFrame:
    path_suffix = suffix or Path(getattr(source, "name", "")).suffix.lower()
    if path_suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if path_suffix == ".json":
        if hasattr(source, "read"):
            data = json.load(source)
        else:
            with Path(source).open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        return pd.DataFrame(data if isinstance(data, list) else data.get("items", []))
    return pd.read_csv(source)


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[\s_]+", " ", str(value).strip().lower())

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_lookup = {_normalize_column_name(col): col for col in df.columns}
    rename_map = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            if alias in normalized_lookup:
                rename_map[normalized_lookup[alias]] = canonical
                break
    result = df.rename(columns=rename_map).copy()
    for column in CANONICAL_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce").fillna(1).astype(int)
    text_columns = [col for col in result.columns if col != "quantity"]
    result[text_columns] = result[text_columns].fillna("").astype(str)
    return result


def load_bom(source: str | Path | BinaryIO, suffix: str | None = None) -> pd.DataFrame:
    """Load a CSV, Excel, or JSON BoM and normalize common fields."""

    df = _read_table(source, suffix)
    return _canonicalize_columns(df)


def load_placement(source: str | Path | BinaryIO, suffix: str | None = None) -> pd.DataFrame:
    """Load KiCad/Altium-style centroid data with best-effort column normalization."""

    df = _read_table(source, suffix)
    aliases = {
        "reference": ["ref", "reference", "designator", "refdes"],
        "x_mm": ["x", "x(mm)", "x pos", "center-x(mm)", "posx"],
        "y_mm": ["y", "y(mm)", "y pos", "center-y(mm)", "posy"],
        "side": ["side", "layer", "top/bottom"],
        "package": ["package", "footprint", "pattern"],
        "rotation": ["rotation", "rot", "angle"],
    }
    normalized_lookup = {_normalize_column_name(col): col for col in df.columns}
    rename_map = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in normalized_lookup:
                rename_map[normalized_lookup[name]] = canonical
                break
    result = df.rename(columns=rename_map).copy()
    for column in aliases:
        if column not in result.columns:
            result[column] = ""
    result["x_mm"] = pd.to_numeric(result["x_mm"], errors="coerce")
    result["y_mm"] = pd.to_numeric(result["y_mm"], errors="coerce")
    return result
