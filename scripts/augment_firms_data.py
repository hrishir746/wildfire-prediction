#!/usr/bin/env python3
"""
Augment FIRMS dataset with rotations and flips to increase training samples.
300 samples → 2400 samples (8x augmentation)
"""
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/firms_progressions")
INPUT_FILE = DATA_DIR / "firms_progressions_256.npz"
OUTPUT_FILE = DATA_DIR / "firms_progressions_augmented.npz"

print("=" * 80)
print("AUGMENT FIRMS DATASET")
print("=" * 80)

# Load original data
print("\n1. Loading original dataset...")
data = np.load(INPUT_FILE)
X = data['X']  # (300, 8, 256, 256)
Y = data['Y']  # (300, 1, 256, 256)

print(f"   Original samples: {len(X)}")
print(f"   Input shape: {X.shape}")
print(f"   Target shape: {Y.shape}")

# Augmentation functions
def rotate_90(x, y):
    """Rotate 90 degrees clockwise."""
    return np.rot90(x, k=1, axes=(2, 3)), np.rot90(y, k=1, axes=(2, 3))

def rotate_180(x, y):
    """Rotate 180 degrees."""
    return np.rot90(x, k=2, axes=(2, 3)), np.rot90(y, k=2, axes=(2, 3))

def rotate_270(x, y):
    """Rotate 270 degrees clockwise."""
    return np.rot90(x, k=3, axes=(2, 3)), np.rot90(y, k=3, axes=(2, 3))

def flip_horizontal(x, y):
    """Flip horizontally."""
    return np.flip(x, axis=3), np.flip(y, axis=3)

def flip_vertical(x, y):
    """Flip vertically."""
    return np.flip(x, axis=2), np.flip(y, axis=2)

def flip_both(x, y):
    """Flip both axes."""
    x_flip = np.flip(x, axis=3)
    x_flip = np.flip(x_flip, axis=2)
    y_flip = np.flip(y, axis=3)
    y_flip = np.flip(y_flip, axis=2)
    return x_flip, y_flip

# Apply augmentations
print("\n2. Applying augmentations...")
augmentations = [
    ("original", lambda x, y: (x, y)),
    ("rot90", rotate_90),
    ("rot180", rotate_180),
    ("rot270", rotate_270),
    ("flip_h", flip_horizontal),
    ("flip_v", flip_vertical),
    ("flip_both", flip_both),
    ("rot90_flip_h", lambda x, y: flip_horizontal(*rotate_90(x, y))),
]

X_aug_list = []
Y_aug_list = []

for name, aug_fn in augmentations:
    print(f"   Applying {name}...")
    X_transformed, Y_transformed = aug_fn(X.copy(), Y.copy())
    X_aug_list.append(X_transformed)
    Y_aug_list.append(Y_transformed)

# Concatenate all augmented data
print("\n3. Combining augmented samples...")
X_aug = np.concatenate(X_aug_list, axis=0)
Y_aug = np.concatenate(Y_aug_list, axis=0)

print(f"   Augmented samples: {len(X_aug)}")
print(f"   Augmentation factor: {len(X_aug) / len(X):.1f}x")

# Verify data integrity
print("\n4. Verifying data integrity...")
for ch in range(8):
    orig_min, orig_max = X[:, ch, :, :].min(), X[:, ch, :, :].max()
    aug_min, aug_max = X_aug[:, ch, :, :].min(), X_aug[:, ch, :, :].max()
    print(f"   Channel {ch}: original [{orig_min:.3f}, {orig_max:.3f}], augmented [{aug_min:.3f}, {aug_max:.3f}]")

# Check fire pixel density
orig_fire_pixels = (Y > 0.5).sum() / Y.size
aug_fire_pixels = (Y_aug > 0.5).sum() / Y_aug.size
print(f"   Fire pixel density: {orig_fire_pixels:.6f} → {aug_fire_pixels:.6f}")

# Save augmented dataset
print(f"\n5. Saving augmented dataset to {OUTPUT_FILE}...")
np.savez_compressed(
    OUTPUT_FILE,
    X=X_aug.astype(np.float32),
    Y=Y_aug.astype(np.float32)
)

print(f"   ✓ Saved {len(X_aug)} augmented samples")
print(f"   File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")

print("\n" + "=" * 80)
print("AUGMENTATION COMPLETE")
print("=" * 80)
print(f"\nDataset increased from {len(X)} to {len(X_aug)} samples ({len(X_aug)//len(X)}x)")
print("Ready to train with augmented data for better generalization.")
