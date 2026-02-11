"""
Evaluate trained U-Net on a held-out test set.

Loads checkpoint, runs on test dataset (different seed), reports IoU, Dice, RoS error.
Saves summary to outputs/evaluation_summary.txt and a small plot to outputs/eval_samples.png.
Run: python scripts/evaluate_model.py [--checkpoint outputs/checkpoints/unet_final.pt]

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

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data import WildfireSyntheticDataset
from src.model import build_unet
from src.evaluation import iou_binary, dice_score, ros_error_cell_based


def load_config_from_ckpt(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("config", {})


def main():
    parser = argparse.ArgumentParser(description="Evaluate wildfire prediction model")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "unet_final.pt")
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123, help="Test set seed (different from train 42)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train first: python scripts/train_model.py")
        return

    config = load_config_from_ckpt(args.checkpoint)
    sim_cfg = config.get("simulation", {})
    model_cfg = config.get("model", {})
    grid = sim_cfg.get("grid_size", [64, 64])
    grid_size = (int(grid[0]), int(grid[1]))
    cell_size_m = float(sim_cfg.get("cell_size_m", 30.0))
    time_step_s = float(sim_cfg.get("time_step_s", 60.0))
    horizon_steps = 10
    in_ch = model_cfg.get("input_channels", 8)
    out_ch = model_cfg.get("output_channels", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_unet(in_channels=in_ch, out_channels=out_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_dataset = WildfireSyntheticDataset(
        num_samples=args.test_samples,
        grid_size=grid_size,
        horizon_steps=horizon_steps,
        seed=args.seed,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    iou_list, dice_list, ros_list = [], [], []
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.sigmoid(logits)
            pred_np = pred.cpu().numpy()
            target_np = y.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p, t = pred_np[i], target_np[i]
                iou_list.append(iou_binary(p, t))
                dice_list.append(dice_score(p, t))
                ros_list.append(ros_error_cell_based(p, t, cell_size_m=cell_size_m, time_steps=horizon_steps, time_step_s=time_step_s))
            all_preds.append(pred_np)
            all_targets.append(target_np)

    iou_mean, iou_std = float(np.mean(iou_list)), float(np.std(iou_list))
    dice_mean, dice_std = float(np.mean(dice_list)), float(np.std(dice_list))
    ros_mean, ros_std = float(np.mean(ros_list)), float(np.std(ros_list))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {args.test_samples} (seed={args.seed})\n")
        f.write(f"Grid: {grid_size}, horizon: {horizon_steps} steps\n\n")
        f.write(f"IoU:    {iou_mean:.4f} +/- {iou_std:.4f}\n")
        f.write(f"Dice:   {dice_mean:.4f} +/- {dice_std:.4f}\n")
        f.write(f"RoS err (mean abs): {ros_mean:.2f} +/- {ros_std:.2f} (area/time proxy)\n")
    print(f"Evaluation summary saved to {summary_path}")
    print(f"  IoU  = {iou_mean:.4f} +/- {iou_std:.4f}")
    print(f"  Dice = {dice_mean:.4f} +/- {dice_std:.4f}")
    print(f"  RoS error (mean) = {ros_mean:.2f}")

    # Save a small visual: 3 sample rows (truth, pred)
    import matplotlib.pyplot as plt
    pred_all = np.concatenate(all_preds, axis=0)
    target_all = np.concatenate(all_targets, axis=0)
    n_show = min(3, pred_all.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes.reshape(-1, 1)
    for i in range(n_show):
        axes[0, i].imshow(target_all[i].squeeze(), cmap="hot", vmin=0, vmax=1)
        axes[0, i].set_title(f"Ground truth (sample {i+1})")
        axes[0, i].axis("off")
        axes[1, i].imshow(pred_all[i].squeeze(), cmap="hot", vmin=0, vmax=1)
        axes[1, i].set_title(f"Model prediction (sample {i+1})")
        axes[1, i].axis("off")
    plt.suptitle("Evaluation samples: Ground truth vs Model prediction")
    plt.tight_layout()
    eval_plot_path = out_dir / "eval_samples.png"
    plt.savefig(eval_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample plot saved to {eval_plot_path}")


if __name__ == "__main__":
    main()
