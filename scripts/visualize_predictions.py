"""
Visualize model predictions vs ground truth and (optionally) Rothermel baseline.

Loads checkpoint, takes a few samples, plots: input (slope/fuel), ground truth,
model prediction, and optional baseline from simulator. Saves to outputs/prediction_comparison.png.
Run: python scripts/visualize_predictions.py [--checkpoint outputs/checkpoints/unet_final.pt]

UTF-8 encoding configured for Windows compatibility.
"""

import argparse
import sys
import os
from pathlib import Path

# UTF-8 encoding fix for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data import WildfireSyntheticDataset
from src.model import build_unet


def load_config_from_ckpt(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("config", {})


def main():
    parser = argparse.ArgumentParser(description="Visualize model vs truth (and baseline)")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "unet_final.pt")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}. Train first: python scripts/train_model.py")
        return

    config = load_config_from_ckpt(args.checkpoint)
    sim_cfg = config.get("simulation", {})
    model_cfg = config.get("model", {})
    grid = sim_cfg.get("grid_size", [64, 64])
    grid_size = (int(grid[0]), int(grid[1]))
    in_ch = model_cfg.get("input_channels", 8)
    out_ch = model_cfg.get("output_channels", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_unet(in_channels=in_ch, out_channels=out_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    dataset = WildfireSyntheticDataset(
        num_samples=args.samples,
        grid_size=grid_size,
        horizon_steps=10,
        seed=args.seed,
    )

    n_show = min(args.samples, 3)
    fig, axes = plt.subplots(n_show, 4, figsize=(14, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_show):
        x, y = dataset[row]
        x_batch = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x_batch)
            pred = torch.sigmoid(logits).cpu().numpy().squeeze()
        target = y.numpy().squeeze()
        # Input channels: 0=slope, 3=fuel, 7=initial fire
        slope = x[0].numpy()
        fuel = x[3].numpy()
        initial_fire = x[7].numpy()

        axes[row, 0].imshow(slope, cmap="terrain")
        axes[row, 0].set_title("Input: slope")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(target, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth (Rothermel)")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(pred, cmap="hot", vmin=0, vmax=1)
        axes[row, 2].set_title("Model prediction")
        axes[row, 2].axis("off")
        axes[row, 3].imshow((pred >= 0.5).astype(float), cmap="hot", vmin=0, vmax=1)
        axes[row, 3].set_title("Model (threshold 0.5)")
        axes[row, 3].axis("off")

    plt.suptitle("Prediction comparison: Input -> Ground truth (simulator) -> Model -> Model (binary)")
    plt.tight_layout()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "prediction_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
