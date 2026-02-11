# Real Data for Wildfire Prediction

This project can use real-world data from **USGS 3DEP** (DEM), **LANDFIRE** (fuels), and **NOAA/NWS** (weather). The pipeline expects an 8-channel input: slope, aspect sin/cos, fuel, wind speed, wind sin/cos, initial fire.

## DEM (elevation → slope, aspect)

- **SRTM (global):** Uses the [elevation](https://pypi.org/project/elevation/) package. No API key.  
  `pip install elevation` then `dem_source="srtm"` in `build_real_input_stack` or `scripts/fetch_real_data.py --dem srtm`.

- **USGS 3DEP (US only):** Uses the [National Map 3DEP ImageServer](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer).  
  `dem_source="3dep"` in code. Requires `rasterio`. No API key.

## Wind (NOAA / NWS)

- **National Weather Service API** ([api.weather.gov](https://api.weather.gov)): No API key. A **User-Agent** header is required.  
  Use `fetch_wind_nws(lat, lon)` or `scripts/fetch_real_data.py --wind-nws` to get current wind speed (m/s) and direction (degrees) for a point.

## LANDFIRE (fuels)

Three options (in priority order):

1. **Local GeoTIFF:** Pre-download from [LANDFIRE](https://landfire.gov/) and pass to `fetch_landfire_fuels(..., local_geotiff=path)`. Most reliable.

2. **WMS (Synchronous, Real Data):** The code automatically fetches LANDFIRE fuel data via USGS WMS for the US (Layer: US_200FBFM40 — Fuel Bed Fuel Model).  
   - No API key required.
   - Called automatically in `build_real_input_stack()` if no local file.
   - Suitable for small-to-medium AOIs.

3. **LFPS (Asynchronous):** [https://lfps.usgs.gov/](https://lfps.usgs.gov/). Provide AOI as **W S E N** (WGS84), layer list, resolution, and email.  
   - Delivers by email or link when ready (hours to days).
   - Best for large areas or high-resolution requests.
   - Can download result and use as local GeoTIFF (option 1).

If all options fail, the code uses a **uniform fuel** placeholder (0.7) so the pipeline still runs. Per SOURCES.md (FlamMap), LANDFIRE is the recommended fuel source for this project methodology.

## Quick run

```bash
# Fetch real DEM + optional NWS wind for a bbox, save 8-channel input and preview
python scripts/fetch_real_data.py --west -122.5 --south 37.5 --east -122 --north 38 --size 64 64 --wind-nws
```

Outputs go to `outputs/real_data/`: `real_input_8ch.npy`, `real_input_preview.png`.

## Using real input with the model

Load the saved array and run the trained U-Net:

```python
import numpy as np
import torch
from src.model import build_unet

x = np.load("outputs/real_data/real_input_8ch.npy")
x = torch.from_numpy(x).float().unsqueeze(0)  # (1, 8, H, W)
model = build_unet(8, 1)
model.load_state_dict(torch.load("outputs/checkpoints/unet_final.pt", weights_only=True)["model_state_dict"])
model.eval()
with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits).numpy().squeeze()
# prob is fire probability map (H, W)
```

## Config

`config/default.yaml` lists sources: `dem_source: "usgs_3dep"`, `fuel_source: "landfire"`, `weather_source: "noaa"`. The code uses these as references; actual fetching is in `src/data/real_data.py` and `scripts/fetch_real_data.py`.
