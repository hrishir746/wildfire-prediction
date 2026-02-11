"""
Comprehensive evaluation pipeline: test trained model against real historical fire data.

Per SOURCES.md (FlamMap, NASA): validation against observed fire perimeters is essential
for assessing model predictions in a "relative sense" for landscape-level comparison.

This script:
1. Loads trained U-Net checkpoint
2. Fetches real DEM, LANDFIRE, and weather data for a historical fire region
3. Loads observed fire perimeter (MTBS or user GeoJSON)
4. Runs inference and compares to observed extent
5. Reports IoU, Dice, RoS error vs baseline

Run: python scripts/evaluate_real_fires.py --event NV3757820190807 --year 2019

UTF-8 encoding configured for Windows compatibility.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, Optional

# UTF-8 encoding fix for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.data import (
    build_real_input_stack,
    load_mtbs_raster,
    fetch_wind_nws,
)
from src.model import build_unet
from src.evaluation import iou_binary, dice_score
from src.simulation import FireSpreadSimulator


# Known event locations (example for testing)
FIRE_EVENTS = {
    "dixie_2021": {
        "bbox": (-121.5, 39.5, -120.5, 40.5),  # W, S, E, N
        "year": 2021,
        "mtbs_id": "CA4017120210808",
    },
    "camp_2018": {
        "bbox": (-121.3, 38.3, -120.8, 38.8),  # California
        "year": 2018,
        "mtbs_id": "CA4004820181108",
    },
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_on_real_fire(
    model_ckpt: Path,
    event_name: str,
    event_bbox: Tuple[float, float, float, float],
    event_year: int,
    mtbs_id: Optional[str] = None,
    size: Tuple[int, int] = (512, 512),  # Default 512x512 for large-scale mapping
    device: torch.device = None,
) -> dict:
    """
    Evaluate model on a real historical fire event.

    Args:
        model_ckpt: Path to trained model checkpoint.
        event_name: Name of fire event.
        event_bbox: Bbox (W, S, E, N) in WGS84.
        event_year: Year of fire.
        mtbs_id: Optional MTBS event ID for downloading perimeter.
        size: Raster size (H, W).
        device: Torch device.

    Returns:
        Dict with evaluation metrics: iou, dice, event_name, bbox, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Evaluating on: {event_name} ({event_year})")
    print(f"Bbox: {event_bbox}")
    print(f"{'='*60}")

    # Load checkpoint and config
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    in_ch = model_cfg.get("input_channels", 8)
    out_ch = model_cfg.get("output_channels", 1)

    # Load model
    model = build_unet(in_channels=in_ch, out_channels=out_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Fetch real data
    print("Fetching real data (DEM, wind, fuels)...")
    west, south, east, north = event_bbox
    try:
        stack = build_real_input_stack(
            west, south, east, north,
            size,
            dem_source="3dep",
            fetch_wind_from_nws=True,
            nws_lat=(south + north) / 2,
            nws_lon=(west + east) / 2,
        )
        print(f"  Input stack shape: {stack.shape}")
    except Exception as e:
        print(f"  Error fetching real data: {e}")
        return {"error": str(e)}

    # Run inference
    print("Running model inference...")
    x = torch.from_numpy(stack).float().unsqueeze(0).to(device)  # (1, 8, H, W)
    with torch.no_grad():
        logits = model(x)
        pred_prob = torch.sigmoid(logits).cpu().numpy().squeeze()  # (H, W)
    pred_binary = (pred_prob > 0.5).astype(np.float32)

    # Load observed fire extent (MTBS)
    print("Loading observed fire extent...")
    obs_extent = None
    if mtbs_id:
        severity = load_mtbs_raster(mtbs_id, event_year)
        if severity is not None:
            obs_extent = (severity > 0).astype(np.float32)
            obs_extent = _resize_if_needed(obs_extent, size)
            print(f"  MTBS severity loaded: {obs_extent.shape}")
        else:
            print(f"  MTBS fetch failed for {mtbs_id}")

    # Also compute baseline (Rothermel simulator) for comparison
    print("Computing Rothermel baseline...")
    baseline_extent = _compute_baseline(stack, size, device)

    # Compute metrics
    results = {
        "event_name": event_name,
        "year": event_year,
        "bbox": event_bbox,
        "model_pred_prob": pred_prob,
        "model_pred_binary": pred_binary,
        "baseline": baseline_extent,
        "observed": obs_extent,
    }

    if obs_extent is not None:
        iou_model = iou_binary(pred_binary, obs_extent)
        dice_model = dice_score(pred_binary, obs_extent)
        iou_baseline = iou_binary(baseline_extent, obs_extent)
        dice_baseline = dice_score(baseline_extent, obs_extent)

        results["iou_model"] = iou_model
        results["dice_model"] = dice_model
        results["iou_baseline"] = iou_baseline
        results["dice_baseline"] = dice_baseline
        results["iou_improvement"] = iou_model - iou_baseline
        results["dice_improvement"] = dice_model - dice_baseline

        print(f"\nMetrics vs observed fire extent:")
        print(f"  Model IoU:     {iou_model:.4f}")
        print(f"  Baseline IoU:  {iou_baseline:.4f}")
        print(f"  Improvement:   {results['iou_improvement']:+.4f}")
        print(f"  Model Dice:    {dice_model:.4f}")
        print(f"  Baseline Dice: {dice_baseline:.4f}")
    else:
        print("  Observed extent not available for comparison.")

    return results


def _resize_if_needed(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize array to target size if needed."""
    if arr.shape == size:
        return arr
    from scipy.ndimage import zoom
    zoom_factors = (size[0] / arr.shape[0], size[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=0)


def _compute_baseline(
    stack: np.ndarray,
    size: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """Compute Rothermel baseline fire extent for comparison."""
    from src.simulation.rothermel_simulator import SimulationParams

    rows, cols = size
    sim_params = SimulationParams(grid_rows=rows, grid_cols=cols, cell_size_m=30.0)
    sim = FireSpreadSimulator(sim_params)

    # Extract data from stack
    slope_norm = stack[0]
    aspect_sin = stack[1]
    aspect_cos = stack[2]
    fuel = stack[3]
    wind_speed = stack[4, 0, 0]  # Uniform grid value
    wind_sin = stack[5, 0, 0]
    wind_cos = stack[6, 0, 0]
    initial_fire = stack[7]

    # Denormalize
    slope_deg = slope_norm * 45.0
    aspect_deg = np.degrees(np.arctan2(-wind_sin, wind_cos)) % 360
    wind_direction_deg = np.degrees(np.arctan2(wind_sin, wind_cos)) % 360
    wind_speed_m_s = wind_speed * 20.0

    # Run simulation
    fire_state = (initial_fire > 0.5).astype(np.int32)
    for _ in range(10):
        if not np.any(fire_state == 1):
            break
        fire_state = sim.step(
            fire_state, slope_deg, aspect_deg,
            wind_speed_m_s, wind_direction_deg, fuel
        )

    return (fire_state > 0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on real historical fires")
    parser.add_argument("--event", type=str, required=True, help="Event name or MTBS ID")
    parser.add_argument("--year", type=int, help="Year of event (if known)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    parser.add_argument("--mtbs-id", type=str, help="MTBS event ID")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "unet_final.pt")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"), help="Patch size (default: 512x512)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "evaluation_real")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train first: python scripts/train_model.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine event details
    event_info = FIRE_EVENTS.get(args.event)
    if event_info:
        bbox = event_info["bbox"]
        year = event_info["year"]
        mtbs_id = event_info.get("mtbs_id")
    else:
        if not args.bbox or not args.year:
            print(f"Event '{args.event}' not in preset list.")
            print("Provide --bbox W S E N and --year")
            return
        bbox = tuple(args.bbox)
        year = args.year
        mtbs_id = args.mtbs_id

    size = tuple(args.size)

    # Evaluate
    results = evaluate_on_real_fire(
        args.checkpoint,
        args.event,
        bbox,
        year,
        mtbs_id=mtbs_id,
        size=size,
        device=device,
    )

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / f"{args.event}_results.npy"
    np.save(results_path, results, allow_pickle=True)
    print(f"\nResults saved to {results_path}")

    # Save visualizations
    if results.get("model_pred_prob") is not None:
        import matplotlib.pyplot as plt
        observed = results.get("observed")
        if observed is not None:
            fig, axes = plt.subplots(1, 4, figsize=(14, 3))
            axes[0].imshow(results["model_pred_prob"], cmap="hot")
            axes[0].set_title("Model Prediction")
            axes[0].axis("off")

            axes[1].imshow(results["baseline"], cmap="hot")
            axes[1].set_title("Rothermel Baseline")
            axes[1].axis("off")

            axes[2].imshow(observed, cmap="hot")
            axes[2].set_title("Observed (MTBS)")
            axes[2].axis("off")

            axes[3].imshow(results["model_pred_prob"] - observed, cmap="RdBu_r")
            axes[3].set_title("Difference (Model - Observed)")
            axes[3].axis("off")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            axes[0].imshow(results["model_pred_prob"], cmap="hot")
            axes[0].set_title("Model Prediction")
            axes[0].axis("off")

            axes[1].imshow(results["baseline"], cmap="hot")
            axes[1].set_title("Rothermel Baseline")
            axes[1].axis("off")

            # Difference vs baseline when observed is not available
            diff = results["model_pred_prob"] - results["baseline"]
            axes[2].imshow(diff, cmap="RdBu_r")
            axes[2].set_title("Difference (Model - Baseline)")
            axes[2].axis("off")

        plt.suptitle(f"{args.event} ({year})")
        plt.tight_layout()
        viz_path = args.out_dir / f"{args.event}_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to {viz_path}")


if __name__ == "__main__":
    main()
