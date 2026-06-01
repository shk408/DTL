from pathlib import Path

from pcb_sustainability.ingestion import load_bom
from pcb_sustainability.scoring import RuleBasedSustainabilityScorer, score_bom


ROOT = Path(__file__).resolve().parents[1]


def test_battery_or_pvc_component_gets_toxicity_warning():
    df = load_bom(ROOT / "samples" / "sample_bom.csv")
    battery_row = df[df["part_number"] == "BAT-HOLDER-CR2032"].iloc[0]
    score = RuleBasedSustainabilityScorer().score_component(battery_row, {})
    labels = {rec.label for rec in score.recommendations}
    assert "High toxicity risk" in labels


def test_bom_summary_score_is_bounded():
    df = load_bom(ROOT / "samples" / "sample_bom.csv")
    scores, summary = score_bom(df)
    assert len(scores) == len(df)
    assert 0 <= summary["summary_score"] <= 100
    assert summary["high_priority_recommendations"] >= 1
