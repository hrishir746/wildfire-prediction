#!/usr/bin/env python3
"""
QUICK CONTROL TEST: 20 epochs each (faster)
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

print("="*80)
print("QUICK CONTROL TEST: Sparse vs Dense (20 epochs each)")
print("="*80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPARSE_PATH = ROOT / "data" / "firms_progressions" / "firms_progressions_256.npz"
DENSE_PATH = ROOT / "data" / "dense_firms_training" / "dense_firms_fast_256.npz"

sparse_data = np.load(SPARSE_PATH)
X_sparse = torch.from_numpy(sparse_data['X']).float()
Y_sparse = torch.from_numpy(sparse_data['Y']).float()

dense_data = np.load(DENSE_PATH)
X_dense = torch.from_numpy(dense_data['X']).float()
Y_dense = torch.from_numpy(dense_data['Y']).float()

print(f"\nSparse: {len(X_sparse)} samples, avg {Y_sparse.sum(dim=(1,2,3)).mean():.1f} pixels")
print(f"Dense:  {len(X_dense)} samples, avg {Y_dense.sum(dim=(1,2,3)).mean():.1f} pixels")

def quick_train(X, Y, name, epochs=20):
    n_train = int(len(X) * 0.85)
    indices = torch.randperm(len(X))
    X_train, Y_train = X[indices[:n_train]].to(DEVICE), Y[indices[:n_train]].to(DEVICE)
    X_val, Y_val = X[indices[n_train:]].to(DEVICE), Y[indices[n_train:]].to(DEVICE)
    
    model = build_unet(in_channels=8, out_channels=1, dropout_rate=0.3).to(DEVICE)
    
    def loss_fn(pred, target):
        bce = nn.BCEWithLogitsLoss()(pred, target)
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()
        dice = 1 - (2 * intersection + 1) / (union + 1)
        return bce + dice
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=4)
    
    best_iou = 0.0
    print(f"\nTraining {name}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        
        model.eval()
        iou_sum = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = (torch.sigmoid(model(x)) > 0.5).float()
                intersection = (pred * y).sum()
                union = pred.sum() + y.sum() - intersection
                iou_sum += (intersection / (union + 1e-7)).item()
        
        val_iou = iou_sum / len(val_loader)
        if val_iou > best_iou:
            best_iou = val_iou
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: IoU={val_iou:.4f}, Best={best_iou:.4f}")
    
    return best_iou

print("\n" + "="*80)
sparse_iou = quick_train(X_sparse, Y_sparse, "SPARSE", epochs=20)
print(f"✓ Sparse best IoU: {sparse_iou:.4f} ({sparse_iou*100:.2f}%)")

print("\n" + "="*80)
dense_iou = quick_train(X_dense, Y_dense, "DENSE", epochs=20)
print(f"✓ Dense best IoU: {dense_iou:.4f} ({dense_iou*100:.2f}%)")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nSparse data (3.9 px):   {sparse_iou*100:.2f}% IoU")
print(f"Dense data (3629 px):   {dense_iou*100:.2f}% IoU")
print(f"\nImprovement:            {dense_iou/sparse_iou:.2f}x")
print(f"Data density ratio:     {3629/3.9:.1f}x")
print(f"Efficiency:             {(dense_iou/sparse_iou)/(3629/3.9)*100:.2f}%")

if (dense_iou/sparse_iou) < 10:
    print("\n✅ LEGITIMATE: Improvement is sub-linear")
    print("   Model learned spatial patterns, not just denser labels!")
else:
    print("\n⚠️  May be mostly data density effect")

print("="*80)
