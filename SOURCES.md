# Project Sources and How They Guide This Work

All methodology, model design, baseline comparison, limitations, and science communication in this project are guided by the following sources. Reasoning and techniques should reference them where applicable.

---

## 1. NASA – Satellite monitoring and environmental data

**Cole, Steve.** “Battling Wildfires from Space: NASA Adds to Firefighters’ Toolkit.” NASA, 16 July 2015.  
https://www.nasa.gov/news-release/battling-wildfires-from-space-nasa-adds-to-firefighters-toolkit/

**Use in this project:**
- **Satellite monitoring:** High-resolution fire detection (e.g., VIIRS 375 m) for active fire locations and progression; supports validation and future integration of real-time fire perimeters.
- **Environmental data integration:** Combining satellite-derived products with weather and land data for fire behavior prediction, consistent with NASA–USFS collaboration described in the source.
- **AI-assisted fire observation:** Framing our neural-network predictions as a decision-support layer that can complement (not replace) operational tools fed by satellite and weather-fire models.

**Project components:** Data layer (future DEM/satellite/weather integration), evaluation (comparing model output to observed or satellite-detected fire extent), README/science communication.

---

## 2. U.S. Forest Service – FlamMap (terrain, vegetation, weather, baseline)

**U.S. Forest Service.** “FlamMap Project.” U.S. Department of Agriculture, 4 Dec. 2025.  
https://research.fs.usda.gov/firelab/projects/flammap

**Use in this project:**
- **Terrain, vegetation, weather inputs:** Our 8-channel input design (slope, aspect, fuel, wind speed/direction) aligns with FlamMap’s use of landscape files under constant or varying environmental conditions (weather, fuel moisture). LANDFIRE fuels are the stated fuel source in our config, consistent with FlamMap’s use of LANDFIRE.
- **Baseline for comparison:** FlamMap implements **Rothermel (1972)** surface fire spread and FARSITE for fire growth. Our Rothermel-inspired simulator is a simplified 2D implementation of the same family of models; we compare AI predictions to this baseline (e.g., IoU, rate-of-spread error) in a relative sense, as FlamMap outputs are intended for “landscape level comparisons” and “relative sense rather than absolute sense.”
- **Limitations:** FlamMap notes that constant-condition runs do not capture temporal variation or backing/flanking behavior; we acknowledge similar limitations for our baseline and AI model.

**Project components:** `config/default.yaml` (data sources), `src/simulation/rothermel_simulator.py`, evaluation metrics and baseline comparison, limitations section in README/SOURCES.

---

## 3. Rothermel (1972) – Deterministic baseline fire spread model

**Rothermel, Richard C.** *A Mathematical Model for Predicting Fire Spread in Wildland Fuels.* U.S. Forest Service Research Paper INT-115, 1972.  
https://research.fs.usda.gov/treesearch/download/32533.pdf

**Use in this project:**
- **Baseline model:** Our simulator implements a **Rothermel-inspired** 2D cellular spread model. Rothermel’s model predicts **rate of spread (R)** and **reaction intensity** from fuel and environment; we use the same physical drivers: **wind** (spread favored downwind), **slope** (spread favored uphill), and **fuel** (loading/surface-area-to-volume conceptually represented by our fuel density channel).
- **Equation structure:** Rothermel gives \(R \propto I_R / (\rho_b E Q_{ig})\) with wind and slope coefficients (\(\phi_w\), \(\phi_s\)). Our spread weights combine wind factor (direction/speed) and slope factor (aspect/slope) in a simplified form consistent with this structure.
- **Inputs:** Rothermel’s inputs include fuel loading, depth, surface-area-to-volume ratio, moisture, wind at midflame height, and slope. Our 8-channel input (slope, aspect, fuel, wind speed, wind direction sin/cos, initial fire) is a reduced set suitable for a neural network while retaining the main physical controls.
- **Limitations (from source):** Model is for **surface fire** in a continuous fuel stratum; no crown fire, no spotting/firebrands; **quasi-steady** spread (line fire). We state the same scope and limitations for our baseline.

**Project components:** `src/simulation/rothermel_simulator.py` (design and docstrings), evaluation (RoS error vs baseline), PROJECT_SPECIFICATION and README (baseline and limitations).

---

## 4. Sullivan (2009) – Limitations of traditional models, justification for AI

**Sullivan, Andrew L.** “Wildland Surface Fire Spread Modelling, 1990–2007.” *International Journal of Wildland Fire*, vol. 18, no. 4, 2009, pp. 349–368.  
https://nrfirescience.org/resource/13825

**Use in this project:**
- **Limitations of traditional models:** Sullivan surveys simulation and mathematical-analogue models that extend empirical/quasi-empirical (often 1D) models to 2D landscape spread. We cite this to explain that many operational tools are **implementations of existing empirical or quasi-empirical models** and that their behavior is constrained by those underlying assumptions. This justifies exploring **AI-based approaches** that can learn from data and capture patterns difficult to encode in closed-form equations.
- **Science communication:** When describing why we add a neural network alongside a Rothermel baseline, we reference Sullivan’s survey: traditional models are valuable but have known limits; our AI component is positioned as complementary and validated against such baselines.

**Project components:** README (rationale for AI), PROJECT_SPECIFICATION (approved references), limitations and “why AI” in documentation.

---

## 5. Kim et al. / ISPRS (2023) – AI model design and validation

**Marjani, M. and Mesgari, M. S.** “Large-Scale Wildfire Spread Prediction Using a Multi-Kernel Convolutional Neural Network.” *ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci.*, X-4/W1-2022, 483–488, 2023.  
https://doi.org/10.5194/isprs-annals-X-4-W1-2022-483-2023  
(Project also draws on **Kim et al. (2023)** “Large-Scale Wildfire Spread Prediction Using a Multi-Kernel CNN” as cited in the specification.)

**Use in this project:**
- **Model design:** Multi-kernel CNN / convolutional approaches that take **elevation, wind direction and speed, temperature, humidity, precipitation, drought index, NDVI, and energy release component** and predict **pixel-level fire** (whether a pixel will be on fire at a future time). Our U-Net uses an 8-channel input (terrain, fuel, wind, initial fire) and outputs a **probability map** per pixel, consistent with this “will this pixel burn?” formulation.
- **Validation:** The reference uses accuracy and F1; we use **IoU** and **Dice** for segmentation and **RoS error** for comparison to the Rothermel baseline, with the same intent of rigorous, quantitative validation.
- **Environmental inputs:** We align our input channels with the idea of real environmental inputs (DEM-derived slope/aspect, fuel, weather) even when training on synthetic data first, so that the pipeline is ready for real DEM/LANDFIRE/NOAA data as in the cited work.

**Project components:** `src/model/unet.py` (architecture and input/output design), `src/data/wildfire_dataset.py` (input stack), evaluation metrics (IoU, F1/Dice, RoS), config `input_channels`/`output_channels`.

---

## Summary table

| Source        | Guides |
|---------------|--------|
| NASA (2015)   | Satellite monitoring, environmental data integration, AI-assisted fire observation framing. |
| FlamMap       | Terrain/fuel/weather inputs, baseline comparison, LANDFIRE, relative (not absolute) interpretation of outputs. |
| Rothermel 72  | Deterministic baseline equations, wind/slope/fuel effects, scope (surface, quasi-steady), limitations. |
| Sullivan 09   | Limitations of traditional models, justification for AI-based approaches. |
| Kim et al. 23 | CNN design, pixel-level prediction, environmental inputs, validation (accuracy/F1; we use IoU, Dice, RoS). |

When you add or change methodology, model design, baseline comparison, or limitations, update the relevant section above and cite the source in code comments or docs where applicable.
