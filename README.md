# Wildfire Spread Prediction System

A scientifically defensible AI system for predicting short-term wildfire spread using real-world environmental data. Designed for PJAS/ISEF and research use. **All methodology, model design, baseline comparison, and limitations are guided by the sources in [SOURCES.md](SOURCES.md).**

## Project Overview

- **Environmental Data Layer:** DEM terrain, LANDFIRE fuels, NOAA weather (aligned with FlamMap and NASA satellite/environmental data integration).
- **Fire Spread Simulation:** Rothermel-inspired deterministic baseline (Rothermel 1972; FlamMap uses the same surface fire spread model).
- **Neural Network:** Convolutional (U-Net) model for pixel-level probability maps (Kim et al. 2023–style; environmental inputs).
- **Conversational Agent:** Educational, read-only interpretive interface.

## Scientific Basis and Sources

| Purpose | Source |
|--------|--------|
| **Baseline model** | Rothermel (1972), U.S. Forest Service — rate of spread and reaction intensity from wind, slope, fuel. |
| **Inputs and framework** | FlamMap (USFS) — terrain, fuels, weather; LANDFIRE; constant or varying conditions. |
| **Satellite / environmental data** | NASA (2015) — VIIRS fire detection, weather–fire models, integration with land/weather data. |
| **Limitations of traditional models** | Sullivan (2009) — simulation models as implementations of empirical/quasi-empirical models; motivates AI. |
| **AI design and validation** | Kim et al. / Marjani & Mesgari (2023) — multi-kernel CNN, pixel-level prediction, environmental inputs, accuracy/F1. |

Full citations and how each source guides this project: **[SOURCES.md](SOURCES.md)**.

## Methodology (source-guided)

- **Baseline:** Our simulator implements a **Rothermel-inspired** 2D cellular model: spread to 8 neighbors with weights from wind (downwind favored) and slope (uphill favored), and fuel density (Rothermel 1972; FlamMap).
- **AI:** U-Net takes 8 channels (slope, aspect sin/cos, fuel, wind speed, wind sin/cos, initial fire) and outputs a fire probability map per pixel, comparable to multi-kernel CNN approaches (Kim et al. 2023).
- **Evaluation:** IoU and Dice for extent; RoS error vs baseline. FlamMap advises interpreting such outputs in a **relative sense** for comparison (e.g., treatment effectiveness), not as absolute forecasts.
- **Limitations:** Rothermel (1972) is for **surface fire**, quasi-steady spread; no crown fire or spotting. FlamMap notes constant-condition runs do not capture full temporal or spatial variation. Our model is approximate and for **educational/research use only** — not for emergency decisions.

## Quick Start

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# NOTE: For Windows users with Rovo Dev extension
# If you encounter UnicodeEncodeError, use the safe wrapper:
# python scripts/run_training_safe.py SCRIPT_NAME
# See UNICODE_FIX_SOLUTION.md for details

# Run deterministic simulator demo (no external data needed)
python scripts/run_simulation_demo.py

# Train neural network with data augmentation and improved loss (synthetic data)
python scripts/train_model.py --epochs 50 --samples 1000

# OR use safe wrapper (recommended for Windows/Rovo Dev):
python scripts/run_training_safe.py train_model --epochs 50 --samples 1000

# Evaluate model (after training)
python scripts/evaluate_model.py

# Evaluate on real historical fires (MTBS data)
python scripts/evaluate_real_fires.py --event camp_2018

# Fetch real environmental data and run inference
python scripts/fetch_real_data.py --west -122.5 --south 37.5 --east -122 --north 38 --wind-nws
python scripts/visualize_predictions.py

# Conversational agent (CLI or web UI)
python scripts/agent_app.py --cli --checkpoint outputs/checkpoints/unet_final.pt
# OR
streamlit run scripts/agent_app.py

# Live training dashboard (NEW!)
# Terminal 1: Start training with dashboard logging
python scripts/train_with_dashboard.py --epochs 50 --samples 1000

# Terminal 2: Launch dashboard
streamlit run scripts/training_dashboard.py

# Integration test (512x512 pipeline validation)
python scripts/test_full_pipeline_512.py
```

## Training Configuration

**Default Patch Size:** All training and validation scripts now use **512x512** patches by default for large-scale wildfire mapping (upgraded from 256x256). This provides:
- Higher spatial resolution for terrain and fuel features
- Better capture of fire spread patterns
- More accurate predictions for real-world fire events

**Batch Sizes:** Adjusted for GPU memory with 512x512 patches:
- Training: 2-4 samples per batch (depending on GPU VRAM)
- Validation: 2-4 samples per batch

**UTF-8 Encoding:** All scripts include proper UTF-8 encoding configuration for Windows compatibility, preventing Unicode errors in console output and file I/O.

**Outputs (in `outputs/`):**
- `simulation_demo.png` — Rothermel simulation (t=0, mid, final)
- `training_curves.png` — Loss and IoU over epochs
- `eval_samples.png` — Ground truth vs model on test samples
- `evaluation_real/` — Real fire evaluation results (model vs observed)
- `real_data/` — Fetched real DEM, LANDFIRE, weather inputs

## Recent Enhancements (Jan-Feb 2026)

✅ **Real Data Integration:** WMS-based LANDFIRE fuel fetching (automatic, no manual download)  
✅ **Historical Fire Data:** MTBS burn severity and NIFC perimeters for validation  
✅ **Model Improvements:** Dropout regularization, data augmentation (spatial + photometric), combined Dice+BCE loss  
✅ **Real Fire Evaluation:** Full pipeline comparing model predictions to observed fire extents (IoU, Dice vs baseline)  
✅ **Conversational Agent:** Interactive Q&A interface (CLI + Streamlit) explaining predictions and limitations  
✅ **Live Training Dashboard:** Real-time monitoring of training metrics, predictions, and checkpoints (Streamlit-based)  
✅ **512x512 Patch Support:** Updated all training and validation scripts for large-scale wildfire mapping (2x resolution increase)  
✅ **UTF-8 Encoding:** Full Windows compatibility with proper UTF-8 encoding across all scripts (no charmap errors)  
✅ **Unicode Fix for Rovo Dev:** Safe wrapper script + ASCII replacements to prevent UnicodeEncodeError in extension child processes  
✅ **Integration Testing:** End-to-end pipeline test script for verifying 512x512 training and validation  

See **[COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md)** for detailed implementation notes.

## Real data (DEM, LANDFIRE, NOAA)

Real-world inputs are supported via `src/data/real_data.py` and `scripts/fetch_real_data.py`:

- **DEM:** SRTM (global, `elevation` package) or USGS 3DEP (US only, REST). → slope, aspect.
- **Wind:** NOAA NWS API (no key; User-Agent required). → wind speed and direction.
- **Fuels:** LANDFIRE (LFPS or local GeoTIFF) or uniform placeholder.

```bash
pip install elevation   # for SRTM DEM
python scripts/fetch_real_data.py --west -122.5 --south 37.5 --east -122 --north 38 --size 64 64 --wind-nws
```

Outputs: `outputs/real_data/real_input_8ch.npy`, `real_input_preview.png`. See **[DATA_README.md](DATA_README.md)** for details and model usage.

## Limitations (source-informed)

- **Rothermel (1972):** Model applies to surface fire in a continuous fuel stratum; no crown fire or firebrands; quasi-steady line fire. Our baseline inherits this scope.
- **FlamMap:** Outputs are suited to “relative sense” comparisons (e.g., fuel treatment); constant conditions do not capture all temporal/spatial variation.
- **Sullivan (2009):** Traditional simulation models have known limits; AI is complementary and must be validated against baselines and data.
- **Operational use:** This system is **approximate** and **not** for operational prediction or emergency decisions. Predictions depend on input data quality and coverage. Educational and research use only.

## License

Research/educational use. Cite the sources in [SOURCES.md](SOURCES.md) when presenting.
