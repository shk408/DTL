import zipfile
from pathlib import Path

from pcb_sustainability.gerber import parse_gerber_zip, score_pcb
from pcb_sustainability.ingestion import load_bom, load_placement
from pcb_sustainability.recycling import estimate_recovery
from pcb_sustainability.scoring import score_bom


ROOT = Path(__file__).resolve().parents[1]


def _make_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "demo_gerbers.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in (ROOT / "samples" / "demo_gerbers").glob("*"):
            archive.write(path, path.name)
    return zip_path


def test_parse_demo_gerbers_extracts_features(tmp_path):
    placement = load_placement(ROOT / "samples" / "sample_placement.csv")
    features = parse_gerber_zip(_make_zip(tmp_path), placement)
    assert features.board_area_mm2 and features.board_area_mm2 > 0
    assert features.layer_count == 2
    assert features.hole_count == 5
    assert features.component_count == 8


def test_recovery_score_is_bounded(tmp_path):
    bom = load_bom(ROOT / "samples" / "sample_bom.csv")
    components, _ = score_bom(bom)
    features = parse_gerber_zip(_make_zip(tmp_path), load_placement(ROOT / "samples" / "sample_placement.csv"))
    pcb_score = score_pcb(features)
    recovery = estimate_recovery(components, features, pcb_score)
    assert 0 <= recovery.final_score <= 100
    assert "copper_g" in recovery.materials
