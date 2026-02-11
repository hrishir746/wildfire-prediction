"""
Fetch real-world data for a bounding box and build 8-channel model input.

DEM: SRTM (elevation package) or USGS 3DEP (US only). Wind: optional from NWS.
Fuels: uniform placeholder or local GeoTIFF. Saves input as .npy and a preview PNG.

Run: python scripts/fetch_real_data.py --west -122.5 --south 37.5 --east -122 --north 38 [--size 64 64] [--wind-nws]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.real_data import build_real_input_stack, fetch_wind_nws


def main():
    parser = argparse.ArgumentParser(description="Fetch real data and build model input")
    parser.add_argument("--west", type=float, required=True, help="Bbox west (lon)")
    parser.add_argument("--south", type=float, required=True, help="Bbox south (lat)")
    parser.add_argument("--east", type=float, required=True, help="Bbox east (lon)")
    parser.add_argument("--north", type=float, required=True, help="Bbox north (lat)")
    parser.add_argument("--size", type=int, nargs=2, default=[64, 64], metavar=("H", "W"))
    parser.add_argument("--dem", choices=["srtm", "3dep"], default="srtm", help="DEM source (srtm=global, 3dep=US)")
    parser.add_argument("--wind-nws", action="store_true", help="Fetch wind from NWS for bbox center")
    parser.add_argument("--wind-speed", type=float, default=None, help="Override wind speed (m/s)")
    parser.add_argument("--wind-dir", type=float, default=None, help="Override wind direction (deg)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "real_data")
    parser.add_argument("--no-plot", action="store_true", help="Do not save preview PNG")
    args = parser.parse_args()

    size = tuple(args.size)
    if args.wind_nws:
        lat = (args.south + args.north) / 2
        lon = (args.west + args.east) / 2
        ws, wd = fetch_wind_nws(lat, lon)
        print(f"Wind from NWS: {ws:.2f} m/s, {wd:.1f}deg")
        wind_speed, wind_direction_deg = ws, wd
    else:
        wind_speed = args.wind_speed if args.wind_speed is not None else 5.0
        wind_direction_deg = args.wind_dir if args.wind_dir is not None else 270.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dem_cache = (args.out_dir / "dem_temp.tif") if args.dem == "srtm" else None
    print("Fetching real data (DEM, wind, fuels) and building 8-channel input...")
    stack = build_real_input_stack(
        args.west, args.south, args.east, args.north,
        size,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction_deg,
        dem_source=args.dem,
        fetch_wind_from_nws=False,
        dem_cache=dem_cache,
    )
    print(f"Stack shape: {stack.shape}")
    out_npy = args.out_dir / "real_input_8ch.npy"
    np.save(out_npy, stack)
    print(f"Saved 8-channel input to {out_npy}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        ch_names = ["slope", "aspect_sin", "aspect_cos", "fuel", "wind_speed", "wind_sin", "wind_cos", "initial_fire"]
        for i, ax in enumerate(axes.flat):
            if i < 8:
                im = ax.imshow(stack[i], cmap="viridis" if i != 7 else "hot")
                ax.set_title(ch_names[i])
                plt.colorbar(im, ax=ax)
            ax.axis("off")
        plt.suptitle("Real data 8-channel input (preview)")
        plt.tight_layout()
        plot_path = args.out_dir / "real_input_preview.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Preview saved to {plot_path}")


if __name__ == "__main__":
    main()
