# Wildfire Prediction System - Project Specification

## Role
Expert research engineer specializing in wildfire behavior modeling, geospatial data processing, and ML. Build a real-world wildfire prediction system for PJAS/ISEF and research.

## Goal
Scientifically defensible AI that predicts short-term wildfire spread (direction and rate) using real environmental data, validated against historical data, with emphasis on realism and transparency.

## Approved References (guide all methodology, model design, baseline, limitations, and science communication)

- **Cole, Steve.** “Battling Wildfires from Space: NASA Adds to Firefighters’ Toolkit.” NASA, 16 July 2015.  
  https://www.nasa.gov/news-release/battling-wildfires-from-space-nasa-adds-to-firefighters-toolkit/  
  → Satellite monitoring, environmental data integration, AI-assisted fire observation.

- **U.S. Forest Service.** “FlamMap Project.” U.S. Department of Agriculture, 4 Dec. 2025.  
  https://research.fs.usda.gov/firelab/projects/flammap  
  → Terrain, vegetation, weather inputs; baseline for comparison (Rothermel surface fire spread, FARSITE).

- **Rothermel, Richard C.** *A Mathematical Model for Predicting Fire Spread in Wildland Fuels.* U.S. Forest Service Research Paper INT-115, 1972.  
  https://research.fs.usda.gov/treesearch/download/32533.pdf  
  → Deterministic baseline fire spread model against which AI predictions are compared.

- **Sullivan, Andrew L.** “Wildland Surface Fire Spread Modelling, 1990–2007.” *International Journal of Wildland Fire*, vol. 18, no. 4, 2009, pp. 349–368.  
  https://nrfirescience.org/resource/13825  
  → Limitations of traditional models; justification for AI-based approaches.

- **Kim et al. (2023)** / **Marjani & Mesgari (2023).** “Large-Scale Wildfire Spread Prediction Using a Multi-Kernel Convolutional Neural Network.” *ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci.*, X-4/W1-2022, 483–488, 2023.  
  https://doi.org/10.5194/isprs-annals-X-4-W1-2022-483-2023  
  → AI model design, neural network architecture, validation with real environmental inputs.

**Detailed use of each source in this project:** see [SOURCES.md](SOURCES.md).

## Constraints
- Computer-based only; no physical experiments
- Public, ethical datasets only
- No human/animal research
- Probabilistic, explainable predictions
- Document limitations; educational use framing

## Architecture
1. **Data:** DEM, LANDFIRE, NOAA (per FlamMap/NASA; see SOURCES.md)
2. **Simulation:** Rothermel-inspired 2D baseline (Rothermel 1972; FlamMap)
3. **Neural Network:** Conv/spatiotemporal → probability maps (Kim et al. 2023–style; pixel-level prediction)
4. **Evaluation:** IoU, RoS error, baseline comparison (FlamMap: “relative sense”)
5. **Agent:** Read-only interpretive, no model control
