from pathlib import Path

from pcb_sustainability.ingestion import load_bom, load_placement


ROOT = Path(__file__).resolve().parents[1]


def test_load_sample_bom_normalizes_columns():
    df = load_bom(ROOT / "samples" / "sample_bom.csv")
    assert {"part_number", "footprint", "quantity", "compliance"}.issubset(df.columns)
    assert df["quantity"].sum() == 23


def test_load_sample_placement_normalizes_columns():
    df = load_placement(ROOT / "samples" / "sample_placement.csv")
    assert {"reference", "x_mm", "y_mm", "package"}.issubset(df.columns)
    assert len(df) == 8
