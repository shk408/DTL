from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcb_sustainability.export import build_json_report, component_csv_bytes, pdf_report_bytes
from pcb_sustainability.gerber import merge_manual_features, parse_gerber_zip, score_pcb
from pcb_sustainability.ingestion import load_bom, load_placement
from pcb_sustainability.ml import apply_ml_predictions, predict_bom, train_predictor_from_csv
from pcb_sustainability.models import PCBFeatures
from pcb_sustainability.recycling import estimate_recovery
from pcb_sustainability.robu import RobuClient
from pcb_sustainability.scoring import component_scores_to_dataframe, score_bom


st.set_page_config(
    page_title="AI-Assisted PCB Sustainability Optimizer",
    page_icon="PCB",
    layout="wide",
)


def score_gauge(label: str, value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2f855a"},
            "steps": [
                {"range": [0, 40], "color": "#fed7d7"},
                {"range": [40, 70], "color": "#fefcbf"},
                {"range": [70, 100], "color": "#c6f6d5"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=12, r=12, t=36, b=8))
    st.plotly_chart(fig, use_container_width=True)


def uploaded_suffix(uploaded_file) -> str:
    return Path(uploaded_file.name).suffix.lower()


st.title("AI-Assisted Design Optimization for PCBs and Sustainable Electronics")
st.caption("Analyze BoM sustainability, PCB design-for-recycling, Robu.in availability, and end-of-life recovery in one demo-ready workflow.")

with st.sidebar:
    st.header("Inputs")
    bom_file = st.file_uploader("BoM file", type=["csv", "xlsx", "xls", "json"])
    gerber_zip = st.file_uploader("Gerber ZIP", type=["zip"])
    placement_file = st.file_uploader("Placement / centroid file", type=["csv", "xlsx", "xls", "json"])
    st.divider()
    online_robu = st.toggle("Enable live Robu.in lookup", value=True)
    browser_robu = st.toggle("Enable browser-based Robu fallback", value=True)
    enrich_limit = st.number_input("Robu lookup row limit", min_value=1, max_value=100, value=20)
    if st.button("Clear Robu lookup cache", use_container_width=True):
        cache_path = ROOT / ".cache" / "robu_results.json"
        if cache_path.exists():
            cache_path.unlink()
        st.success("Robu lookup cache cleared.")
    if not online_robu:
        st.warning("Live Robu lookup is off. Availability and price will use offline fallback data.")
    elif browser_robu:
        st.caption("Browser fallback will be used only when direct and reader-based Robu lookup fails.")
    st.divider()
    st.subheader("ML scoring")
    enable_ml = st.toggle("Enable ML-assisted scoring", value=False)
    ml_blend = st.slider("ML blend weight", min_value=0.0, max_value=0.5, value=0.25, step=0.05)
    ml_training_file = st.file_uploader("Optional ML training CSV", type=["csv"])
    st.caption("Training CSV must include component fields and `target_score` from 0 to 100.")
    st.divider()
    st.subheader("Manual PCB fallback")
    manual = {
        "board_width_mm": st.number_input("Board width (mm)", min_value=0.0, value=80.0),
        "board_height_mm": st.number_input("Board height (mm)", min_value=0.0, value=50.0),
        "layer_count": st.number_input("Layer count", min_value=0, value=2),
        "hole_count": st.number_input("Hole count", min_value=0, value=70),
        "via_count": st.number_input("Via count", min_value=0, value=45),
        "component_count": st.number_input("Component count", min_value=0, value=25),
        "smd_count": st.number_input("SMD count", min_value=0, value=18),
        "through_hole_count": st.number_input("Through-hole count", min_value=0, value=7),
        "connector_count": st.number_input("Connector count", min_value=0, value=3),
        "edge_component_count": st.number_input("Edge components", min_value=0, value=4),
        "battery_count": st.number_input("Battery count", min_value=0, value=0),
    }

run = st.button("Analyze design", type="primary", use_container_width=True)

if not run:
    st.info("Upload a BoM and optionally a Gerber ZIP or placement file, then run the analysis. Sample files are included in the `samples` folder.")
    st.stop()

if not bom_file:
    st.error("Please upload a BoM file to start the analysis.")
    st.stop()

try:
    bom_df = load_bom(bom_file, uploaded_suffix(bom_file))
except Exception as exc:
    st.error(f"Could not parse BoM: {exc}")
    st.stop()

placement_df = None
if placement_file:
    try:
        placement_df = load_placement(placement_file, uploaded_suffix(placement_file))
    except Exception as exc:
        st.warning(f"Placement file could not be parsed: {exc}")

with st.spinner("Enriching components and scoring sustainability..."):
    client = RobuClient(browser_fallback=browser_robu)
    enrichments = client.enrich_bom(bom_df, enabled=online_robu, limit=int(enrich_limit))
    component_scores, bom_summary = score_bom(bom_df, enrichments)
    ml_error = None
    if enable_ml:
        try:
            training_source = ml_training_file or (ROOT / "samples" / "ml_training_components.csv")
            predictor = train_predictor_from_csv(training_source)
            predictions = predict_bom(bom_df, predictor, enrichments)
            component_scores, bom_summary = apply_ml_predictions(component_scores, predictions, ml_blend)
        except Exception as exc:
            ml_error = str(exc)

if gerber_zip:
    try:
        pcb_features = parse_gerber_zip(gerber_zip, placement_df)
    except Exception as exc:
        pcb_features = PCBFeatures(warnings=[f"Gerber ZIP could not be parsed: {exc}"])
else:
    pcb_features = PCBFeatures(warnings=["No Gerber ZIP uploaded; PCB score uses manual fallback values."])

pcb_features = merge_manual_features(pcb_features, manual)
pcb_score = score_pcb(pcb_features)
recovery = estimate_recovery(component_scores, pcb_features, pcb_score)
report = build_json_report(component_scores, bom_summary, pcb_features, pcb_score, recovery)
if enable_ml and ml_error:
    st.warning(f"ML scoring was skipped: {ml_error}")
elif enable_ml:
    st.success(f"ML-assisted scoring enabled. Rule scores are blended with ML predictions at {int(ml_blend * 100)}% ML weight.")

top_cols = st.columns(3)
with top_cols[0]:
    score_gauge("BoM Sustainability", bom_summary["summary_score"])
with top_cols[1]:
    score_gauge("PCB Recycling Design", pcb_score.score)
with top_cols[2]:
    score_gauge("Final Recyclability", recovery.final_score)

st.subheader("Component Sustainability Report")
component_df = component_scores_to_dataframe(component_scores)
if "robu_status" in component_df.columns:
    unavailable_statuses = {"offline_fallback", "lookup_unavailable", "network_error", "missing_query"}
    statuses = set(component_df["robu_status"].dropna().astype(str))
    if statuses and statuses.issubset(unavailable_statuses):
        st.warning(
            "Robu enrichment did not return live product data for this run. "
            "Make sure live lookup is enabled, clear the Robu cache, or add exact Robu product links "
            "in a `supplier_url` / `robu_url` / `product url` BoM column for reliable parsing."
        )
st.dataframe(component_df, use_container_width=True, hide_index=True)

st.subheader("PCB Layout Features")
feature_cols = st.columns(4)
feature_cols[0].metric("Board area", f"{pcb_features.board_area_mm2 or 0:.1f} mm2")
feature_cols[1].metric("Layers", pcb_features.layer_count)
feature_cols[2].metric("Hole / via count", f"{pcb_features.hole_count} / {pcb_features.via_count}")
feature_cols[3].metric("SMD ratio", f"{pcb_features.smd_ratio:.0%}")
if pcb_features.warnings:
    for warning in pcb_features.warnings:
        st.warning(warning)

st.subheader("Recommendations and Explainability")
left, right = st.columns(2)
with left:
    st.markdown("**Component-level warnings**")
    for item in component_scores:
        for rec in item.recommendations[:2]:
            st.write(f"**{item.part_number}** | {rec.label} | Priority: {rec.priority} | Confidence: {rec.confidence}")
            st.caption(rec.reason)
with right:
    st.markdown("**PCB and recovery warnings**")
    for rec in pcb_score.recommendations:
        st.write(f"**{rec.label}** | Priority: {rec.priority} | Confidence: {rec.confidence}")
        st.caption(rec.reason)
    st.markdown("**Recovery factors**")
    for factor in recovery.positive_factors:
        st.success(factor)
    for factor in recovery.negative_factors:
        st.error(factor)

st.subheader("Recoverable Material Estimate")
materials_df = pd.DataFrame({"material": recovery.materials.keys(), "estimated_value": recovery.materials.values()})
st.dataframe(materials_df, use_container_width=True, hide_index=True)

st.subheader("Exports")
download_cols = st.columns(3)
download_cols[0].download_button("Download CSV", component_csv_bytes(component_scores), "component_sustainability_report.csv", "text/csv")
download_cols[1].download_button("Download JSON", json.dumps(report, indent=2).encode("utf-8"), "pcb_sustainability_report.json", "application/json")
download_cols[2].download_button("Download PDF", pdf_report_bytes(report), "pcb_sustainability_report.pdf", "application/pdf")
