import json
import os
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

# Initialize and inject environment variables from the app folder's .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(
    page_title="AI-Assisted PCB Sustainability Optimizer",
    page_icon="♻️",
    layout="wide",
)

# Local approximate unit-cost model in INR. Tune these numbers for your lab/vendor.
COMPONENT_PRICE_RULES = [
    {
        "category": "Microcontroller / SoC",
        "patterns": ["stm32", "atmega", "attiny", "pic", "nrf52", "rp2040", "esp32", "esp8266", "mcu", "microcontroller"],
        "price": 260.00,
        "confidence": "Medium",
    },
    {
        "category": "Power IC / Regulator",
        "patterns": ["ldo", "regulator", "buck", "boost", "ams1117", "lm2596", "mp1584", "tps", "xl6009"],
        "price": 65.00,
        "confidence": "Medium",
    },
    {
        "category": "General IC",
        "patterns": ["opamp", "op-amp", "comparator", "logic", "driver", "lm358", "ne555", "74hc", "74ls", "max232", "ic"],
        "price": 55.00,
        "confidence": "Medium",
    },
    {
        "category": "Sensor / Module",
        "patterns": ["sensor", "imu", "accelerometer", "gyro", "bme", "bmp", "dht", "mpu", "module"],
        "price": 140.00,
        "confidence": "Low",
    },
    {
        "category": "Connector / Header",
        "patterns": ["conn", "connector", "header", "jst", "usb", "terminal", "screw terminal", "pin header"],
        "price": 28.00,
        "confidence": "Medium",
    },
    {
        "category": "Switch / Button",
        "patterns": ["switch", "button", "pushbutton", "tactile", "tact"],
        "price": 12.00,
        "confidence": "Medium",
    },
    {
        "category": "Crystal / Oscillator",
        "patterns": ["crystal", "xtal", "oscillator", "mhz"],
        "price": 18.00,
        "confidence": "Medium",
    },
    {
        "category": "Inductor / Ferrite",
        "patterns": ["inductor", "ferrite", "bead", "coil", " uh", " nh"],
        "price": 16.00,
        "confidence": "Medium",
    },
    {
        "category": "Diode / LED",
        "patterns": ["diode", "led", "schottky", "zener", "1n4148", "1n4007", "ss14", "bat54"],
        "price": 4.00,
        "confidence": "Medium",
    },
    {
        "category": "Transistor / MOSFET",
        "patterns": ["transistor", "mosfet", "bjt", "2n7002", "bc547", "s8050", "ao3400"],
        "price": 7.00,
        "confidence": "Medium",
    },
    {
        "category": "Capacitor",
        "patterns": ["capacitor", "cap", " c0g", " x7r", " x5r", "uf", "nf", "pf"],
        "price": 2.50,
        "confidence": "Medium",
    },
    {
        "category": "Resistor",
        "patterns": ["resistor", "res", " ohm", "kohm", "k ", "r "],
        "price": 1.20,
        "confidence": "Medium",
    },
]


@st.cache_resource
def load_vision_model():
    try:
        return YOLO("best.pt")
    except Exception:
        return YOLO("yolov8n.pt")


vision_model = load_vision_model()


def _to_float(value):
    if value is None:
        return None
    match = re.search(r"[\d.]+", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def parse_quantity(value):
    number = _to_float(value)
    return max(1, int(number)) if number is not None else 1


def parse_price_inr(value):
    price = _to_float(value)
    return price if price is not None and price > 0 else None


def row_search_text(row):
    return " ".join(str(value) for value in row.values if pd.notna(value)).lower()


def footprint_price_multiplier(search_text):
    if any(size in search_text for size in ["0201", "0402"]):
        return 1.20
    if any(size in search_text for size in ["0603", "0805", "1206"]):
        return 1.00
    if any(term in search_text for term in ["through", "tht", "dip", "to-220", "electrolytic"]):
        return 2.00
    return 1.00


def estimate_local_bom_price(row):
    search_text = row_search_text(row)

    for rule in COMPONENT_PRICE_RULES:
        if any(pattern in search_text for pattern in rule["patterns"]):
            unit_price = rule["price"] * footprint_price_multiplier(search_text)
            return unit_price, rule["category"], f"Local estimate ({rule['confidence']} confidence)"

    return 35.00, "Unclassified Component", "Local estimate (Low confidence)"


def score_gauge(label: str, value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
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


st.title("AI-Assisted Design Optimization for PCBs and Sustainable Electronics")
st.caption(
    "Analyze BOM sustainability, PCB computer vision layout architecture, board complexity, and estimated component costs.")

# --- SIDEBAR DATA INGESTION ---
st.sidebar.header("📁 Data Ingestion Pipeline")
pcb_image = st.sidebar.file_uploader("Upload PCB Layout Image / 3D View", type=["jpg", "jpeg", "png", "webp"])

cv_comp_count = 25
cv_smd_count = 18
cv_th_count = 7
cv_conn_count = 3
cv_annotated_img = None

if pcb_image:
    file_bytes = np.asarray(bytearray(pcb_image.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = vision_model(opencv_img)
    cv_annotated_img = cv2.cvtColor(results[0].plot(conf=0.10, line_width=2), cv2.COLOR_BGR2RGB)

    detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
    if len(detected_classes) > 0:
        cv_comp_count = len(detected_classes)
        cv_smd_count = int(np.sum((detected_classes == 1) | (detected_classes == 2)))
        cv_conn_count = int(np.sum(detected_classes == 5))
        cv_th_count = max(0, cv_comp_count - cv_smd_count)

with st.sidebar:
    bom_file = st.file_uploader("Upload BOM File (CSV / XLSX)", type=["csv", "xlsx"])
    st.divider()

    st.subheader("⚙️ PCB Layout Geometry Parameters")
    manual = {
        "board_width_mm": st.number_input("Board width (mm)", min_value=1.0, value=80.0),
        "board_height_mm": st.number_input("Board height (mm)", min_value=1.0, value=50.0),
        "layer_count": st.number_input("Layer count", min_value=1, max_value=16, value=2),
        "hole_count": st.number_input("Drill hole count", min_value=0, value=35),
        "via_count": st.number_input("Via count", min_value=0, value=24),
        "component_count": st.number_input("Total Component Count", min_value=1, value=cv_comp_count),
        "smd_count": st.number_input("SMD Component Count", min_value=0, value=cv_smd_count),
        "through_hole_count": st.number_input("Through-Hole Count", min_value=0, value=cv_th_count),
        "connector_count": st.number_input("Input Connector Count", min_value=0, value=cv_conn_count),
    }

run = st.button("Run Design Optimization Analysis", type="primary", use_container_width=True)

if not run:
    st.info("💡 Complete sidebar parameters and upload project files to run assessment.")
    st.stop()

if not bom_file:
    st.error("❌ Please upload a BOM file to initiate scoring workflows.")
    st.stop()

# --- BOARD GEOMETRY CALCULATION ---
board_area = manual["board_width_mm"] * manual["board_height_mm"]

# --- SCORING & INTEGRATION ENGINE ---
with st.spinner("Executing calculations and estimating BOM component costs locally..."):
    if bom_file.name.endswith('.csv'):
        bom_df = pd.read_csv(bom_file)
    else:
        bom_df = pd.read_excel(bom_file)

    bom_df.columns = [c.strip().lower() for c in bom_df.columns]
    num_rows = len(bom_df)

    part_col = next(
        (c for c in bom_df.columns if 'part' in c or 'designator' in c or 'name' in c or 'item' in c or 'mpn' in c),
        None)
    qty_col = next((c for c in bom_df.columns if 'qty' in c or 'quantity' in c or 'count' in c), None)
    unit_price_col = next(
        (
            c for c in bom_df.columns
            if ('price' in c or 'cost' in c or 'rate' in c)
            and 'total' not in c
            and 'extended' not in c
        ),
        None,
    )

    if not part_col:
        bom_df['part_number'] = [f"Component_{i + 1}" for i in range(num_rows)]
        part_col = 'part_number'
    if not qty_col:
        bom_df['quantity'] = [1] * num_rows
        qty_col = 'quantity'

    prices_inr = []
    total_costs_inr = []
    pricing_sources = []
    component_categories = []

    for index, row in bom_df.iterrows():
        qty = parse_quantity(row[qty_col])
        unit_price_inr = parse_price_inr(row[unit_price_col]) if unit_price_col else None

        if unit_price_inr is not None:
            component_category = "BOM Supplied Price"
            pricing_source = f"Used uploaded BOM column: {unit_price_col}"
        else:
            unit_price_inr, component_category, pricing_source = estimate_local_bom_price(row)

        prices_inr.append(unit_price_inr)
        total_costs_inr.append(unit_price_inr * qty)
        pricing_sources.append(pricing_source)
        component_categories.append(component_category)

    bom_df['Estimated Category'] = component_categories
    bom_df['Unit Price (INR)'] = [f"₹{p:.2f}" for p in prices_inr]
    bom_df['Total Price (INR)'] = total_costs_inr
    bom_df['Estimate Source'] = pricing_sources
    total_bom_market_cost = sum(total_costs_inr)

    smd_ratio = manual["smd_count"] / manual["component_count"] if manual["component_count"] > 0 else 0.5

    bom_score = 95.0 - (len(bom_df) * 1.5)
    bom_score = max(40.0, min(100.0, bom_score))

    pcb_score = 100.0 - (smd_ratio * 20.0) - (manual["layer_count"] * 8.0) - (manual["via_count"] * 0.2)
    pcb_score = max(10.0, min(100.0, pcb_score))

    final_recyclability = (bom_score * 0.4) + (pcb_score * 0.6)

# --- VISUALIZATION LAYOUT RENDER ---
if cv_annotated_img is not None:
    st.subheader("📸 Computer Vision Architecture Mapping")
    st.image(cv_annotated_img, caption="Trained AI Object Localization Layer (Confidence Threshold: >10%)",
             use_container_width=True)
    st.markdown("---")

top_cols = st.columns(3)
with top_cols[0]:
    score_gauge("BOM Sustainability Score", bom_score)
with top_cols[1]:
    score_gauge("PCB Recycling Profile", pcb_score)
with top_cols[2]:
    score_gauge("Final Circular Recyclability", final_recyclability)

st.subheader("📊 BOM Cost Estimate Report")
st.dataframe(bom_df, use_container_width=True, hide_index=True)
st.metric(label="Total Estimated BOM Cost", value=f"₹{total_bom_market_cost:,.2f}")

st.subheader("📐 Board Complexity Indicators")
feature_cols = st.columns(3)
feature_cols[0].metric("Layer Target Configuration", f"{int(manual['layer_count'])} Layers")
feature_cols[1].metric("Interconnect Structure", f"{int(manual['hole_count'])} / {int(manual['via_count'])}")
feature_cols[2].metric("Surface Density Metric", f"{smd_ratio:.0%}")

st.subheader("💡 Optimization Recommendations & Explainability")
left, right = st.columns(2)
with left:
    st.markdown("**BOM Subassembly Constraints**")
    st.warning(
        "⚠️ **Lead-Free Package Enforcement:** Ensure components match RoHS compliance standards to simplify metallurgical refining separation.")
    st.success(
        "✅ **Cost Estimate Complete:** BOM line items were classified locally and priced using approximate component-category rates.")
with right:
    st.markdown("**Substrate Geometric & Reclamation Warnings**")

    if smd_ratio > 0.70:
        st.error(
            f"❌ **High Surface-Mount Footprint Density ({smd_ratio:.1%}):** Increases automated desoldering thermal dwell time at end-of-life sorting facilities.")

    if manual["layer_count"] > 2:
        st.error(
            "❌ **Substrate Layer Complication:** Multi-layer counts greater than 2 restrict mechanical slicing and copper foil peeling efficiencies.")
