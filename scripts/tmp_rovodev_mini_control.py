#!/usr/bin/env python3
"""
MINIMAL CONTROL: Just 10 epochs to get quick answer
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.model import build_unet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
sparse = np.load(ROOT / "data" / "firms_progressions" / "firms_progressions_256.npz")
dense = np.load(ROOT / "data" / "dense_firms_training" / "dense_firms_fast_256.npz")

X_s, Y_s = torch.from_numpy(sparse['X']).float(), torch.from_numpy(sparse['Y']).float()
X_d, Y_d = torch.from_numpy(dense['X']).float(), torch.from_numpy(dense['Y']).float()

def train_quick(X, Y, epochs=10):
    n = int(len(X)*0.8)
    idx = torch.randperm(len(X))
    Xt, Yt = X[idx[:n]].to(DEVICE), Y[idx[:n]].to(DEVICE)
    Xv, Yv = X[idx[n:]].to(DEVICE), Y[idx[n:]].to(DEVICE)
    
    model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best = 0
    for e in range(epochs):
        model.train()
        for i in range(0, len(Xt), 8):
            x, y = Xt[i:i+8], Yt[i:i+8]
            opt.zero_grad()
            nn.BCEWithLogitsLoss()(model(x), y).backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(Xv)) > 0.5).float()
            inter = (pred * Yv).sum()
            union = pred.sum() + Yv.sum() - inter
            iou = (inter / (union + 1e-7)).item()
            if iou > best:
                best = iou
    
    return best

print("Training on SPARSE (3.9 px/sample)...")
s_iou = train_quick(X_s, Y_s, 10)
print(f"  Best IoU: {s_iou:.4f} ({s_iou*100:.2f}%)")

print("\nTraining on DENSE (3629 px/sample)...")
d_iou = train_quick(X_d, Y_d, 10)
print(f"  Best IoU: {d_iou:.4f} ({d_iou*100:.2f}%)")

print(f"\n{'='*60}")
print(f"SPARSE: {s_iou*100:.2f}%  |  DENSE: {d_iou*100:.2f}%  |  Improvement: {d_iou/s_iou:.2f}x")
print(f"Data density: {3629/3.9:.0f}x  |  Efficiency: {(d_iou/s_iou)/(3629/3.9)*100:.2f}%")

if d_iou/s_iou < 10:
    print(f"\n✅ Sub-linear improvement → Model learned patterns!")
else:
    print(f"\n⚠️ Near-linear → Mostly data density effect")
print(f"{'='*60}")
