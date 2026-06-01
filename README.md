# AI-Assisted Design Optimization for PCBs and Sustainable Electronics

This is a final-year engineering project prototype for evaluating PCB and electronics sustainability during the design phase. It analyzes a Bill of Materials, parses Gerber ZIP packages, optionally enriches components from Robu.in, estimates end-of-life material recovery, and produces an explainable recyclability score from 0 to 100.

## Features

- BoM ingestion from CSV, Excel, or JSON.
- Component-level scoring for toxicity, recyclability, repairability, restricted substance risk, sourcing availability, obsolescence, and greener alternatives.
- Gerber ZIP analysis for board size, layer count, copper estimate, drill count, via density, component density, SMD ratio, edge placement, and disassembly difficulty.
- Optional placement/centroid file support.
- Robu.in enrichment layer with fuzzy search query creation, local cache, offline fallback, availability metadata, and a modular client design for future suppliers.
- Optional ML-assisted component scoring using a trainable TF-IDF regression model.
- End-of-life recovery estimate for copper, contacts, solder, FR4, plastics, metals, and batteries.
- Streamlit UI with uploads, manual PCB fallback fields, explanations, and CSV/JSON/PDF exports.
- Unit tests for ingestion, scoring, Gerber parsing, and recovery estimation.

## Project Structure

```text
.
├── app.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── samples/
│   ├── sample_bom.csv
│   ├── sample_placement.csv
│   └── demo_gerbers/
├── src/pcb_sustainability/
│   ├── export.py
│   ├── gerber.py
│   ├── ingestion.py
│   ├── models.py
│   ├── recommendations.py
│   ├── recycling.py
│   ├── robu.py
│   ├── scoring.py
│   └── utils.py
└── tests/
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the UI:

```powershell
streamlit run app.py
```

Run tests:

```powershell
pytest
```

## GitHub and Streamlit Cloud Deployment

GitHub Pages cannot directly host this app because it is a Python Streamlit application, not a static HTML site. Use GitHub to host the source code, then deploy the app with Streamlit Community Cloud.

Recommended steps:

1. Create a new GitHub repository.
2. Upload these project files and folders: `app.py`, `README.md`, `requirements.txt`, `pyproject.toml`, `runtime.txt`, `.streamlit/`, `src/`, `samples/`, and `tests/`.
3. Do not upload `.venv/`, `.cache/`, `.codex_deps/`, `.pytest_cache/`, `outputs/`, or `__pycache__/`.
4. Go to Streamlit Community Cloud and create a new app from the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.

Live Robu.in lookup may depend on the hosting provider's outbound network policy. The app still works with offline fallback enrichment when live lookup is disabled.

Browser-based fallback is optional and uses Playwright when installed. If you want to use it locally, install the package and browser runtime first:

```powershell
pip install playwright
playwright install chromium
```

The app will automatically continue with non-browser fallback paths when Playwright is unavailable.

To create the sample Gerber ZIP for upload:

```powershell
Compress-Archive -Path samples\demo_gerbers\* -DestinationPath samples\demo_gerbers.zip -Force
```

## Example Usage

1. Start the Streamlit app.
2. Upload `samples/sample_bom.csv`.
3. Upload `samples/sample_placement.csv`.
4. Create and upload `samples/demo_gerbers.zip`, or use manual PCB fallback values.
5. Keep live Robu.in lookup disabled for offline demos, or enable it when internet access is available.
6. Click **Analyze design** and export the CSV, JSON, or PDF report.

## Scoring Logic

The prototype uses deterministic heuristics first, because this makes the demo explainable and repeatable.

The BoM score combines:

- toxicity risk: hazardous keywords such as lead, cadmium, mercury, brominated materials, PVC, and batteries;
- recyclability: modular connectors, copper/steel/aluminum, through-hole serviceability;
- repairability: package difficulty, with BGA/QFN/0201/glue/potting scored lower;
- restricted substance risk: hazardous terms reduced when RoHS, lead-free, halogen-free, or REACH appears;
- sourcing availability: improved by Robu.in availability metadata;
- obsolescence risk: terms such as obsolete, EOL, discontinued, and NRND;
- greener alternatives: compliance and safer-material signals.

## ML-Assisted Scoring

The app includes an optional supervised ML component for component-level sustainability prediction. It trains a TF-IDF + Ridge regression model from a CSV dataset with BoM-like fields and a `target_score` column from 0 to 100.

Default demo training data is included at `samples/ml_training_components.csv`. In the UI, enable **ML-assisted scoring** to train from this sample file, or upload your own training CSV. The ML prediction is blended with the rule-based score using the sidebar blend weight.

The ML output remains explainable:

- predicted sustainability score;
- confidence estimate based on feature coverage;
- risk band: high risk, moderate risk, or lower risk;
- influential terms found in the BoM/enrichment text.

This is intentionally lightweight. It demonstrates where trained models fit into the architecture while preserving deterministic rule behavior for academic evaluation.

The PCB design-for-recycling score combines:

- disassembly difficulty from layer count, via density, SMD ratio, component density, and battery presence;
- accessibility from edge placement and connector count;
- modularity from connectors and edge-serviceable parts;
- material recovery from copper estimate, layer count, and battery penalty.

The final recyclability score combines component sustainability, PCB design-for-recycling, and estimated end-of-life recovery.

## Robu.in Integration

`RobuClient` builds a search query from each BoM row, checks `.cache/robu_results.json`, and only calls Robu.in when live lookup is enabled. It extracts the best available title, category, manufacturer guess, package guess, datasheet link, availability text, price, similar component hints, source URL, and match confidence.

Robu.in may return bot-protection pages to automated Python requests. The integration therefore uses a multi-step lookup:

- If the BoM contains a `supplier_url`, `product url`, `robu url`, `url`, or `link` column with a Robu product URL, parse that product page directly.
- Otherwise try direct Robu.in search using the part number, manufacturer, description, value, and footprint/package fields.
- Search terms are expanded for both SMD and through-hole parts, including 0603/0805/QFN/SOT/BGA, THT, through-hole, DIP, axial, radial, pin headers, JST connectors, and terminal blocks.
- If Robu blocks direct search, try a public Robu page reader.
- If search pages still do not expose product links, use a Robu-scoped URL discovery fallback (`site:robu.in/product ...`) and then parse the discovered Robu product page.
- If all live discovery fails, return structured fallback data, with match confidence and status labels.

This makes arbitrary component lookup best-effort without pretending there is a guaranteed official API. For the most reliable results, include exact Robu product URLs in the BoM when available. The supplier layer is isolated so DigiKey, Mouser, LCSC, Octopart, or a formal API can be added later.

## ML Extension Points

`SustainabilityModel` in `scoring.py` defines the interface a future trained model can implement. The current `RuleBasedSustainabilityScorer` can be replaced or blended with a classifier, ranking model, embedding search system, or LLM-assisted alternatives recommender. The `ml.py` module already demonstrates this pattern with a trainable text regression model.

## Notes

The Gerber parser is a practical prototype parser, not a certified CAM validation tool. It extracts measurable features from common RS-274X and Excellon text files and reports warnings when data is missing. This is intentional: the tool should be honest about uncertainty instead of faking design intelligence.


## Fast Mode Changes
- Removed Playwright from default dependencies.
- Removed Plotly from default dependencies.
- Intended for cached/offline supplier enrichment and coursework demos.
