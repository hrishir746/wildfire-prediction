"""
Train U-Net for wildfire spread prediction on synthetic data.

Uses config/default.yaml. Saves checkpoints and logs to outputs/.
Run: python scripts/train_model.py [--config config/default.yaml]

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
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import WildfireSyntheticDataset
from src.model import build_unet
from src.evaluation import iou_batch_torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice coefficient loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined Dice + BCE loss."""
    bce = nn.BCEWithLogitsLoss()(logits, target)
    dice = dice_loss(logits, target)
    return 0.5 * bce + 0.5 * dice


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train wildfire prediction U-Net")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config max_epochs")
    parser.add_argument("--samples", type=int, default=None, help="Override dataset size")
    args = parser.parse_args()

    config = load_config(args.config)
    sim_cfg = config.get("simulation", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    grid = sim_cfg.get("grid_size", [64, 64])
    grid_size = (int(grid[0]), int(grid[1]))
    horizon = 10
    num_samples = args.samples or 800
    batch_size = model_cfg.get("batch_size", 16)
    lr = model_cfg.get("learning_rate", 1e-4)
    epochs = args.epochs or model_cfg.get("max_epochs", 20)
    in_ch = model_cfg.get("input_channels", 8)
    out_ch = model_cfg.get("output_channels", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Grid: {grid_size}, samples: {num_samples}, batch: {batch_size}, lr: {lr}, epochs: {epochs}")

    train_dataset = WildfireSyntheticDataset(
        num_samples=num_samples,
        grid_size=grid_size,
        horizon_steps=horizon,
        seed=42,
        augment_spatial=True,
        augment_photometric=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_unet(in_channels=in_ch, out_channels=out_ch, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = combined_loss

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    epoch_losses, epoch_ious = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_iou = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                pred = torch.sigmoid(logits)
                iou = iou_batch_torch(pred, y).mean().item()
            total_iou += iou
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        avg_iou = total_iou / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        epoch_ious.append(avg_iou)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  IoU={avg_iou:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            ckpt_path = ckpt_dir / f"unet_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    final_path = ckpt_dir / "unet_final.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, final_path)
    print(f"Training done. Final checkpoint: {final_path}")

    # Save training curves
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epoch_losses, color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE)")
    ax1.set_title("Training loss")
    ax2.plot(epoch_ious, color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("Training IoU")
    plt.tight_layout()
    curves_path = out_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()
