import torch

ckpt = torch.load('outputs/checkpoints/unet_firms_augmented.pt', map_location='cpu')
print(f"Saved checkpoint:")
print(f"  Epoch: {ckpt['epoch']}")
print(f"  IoU: {ckpt['iou']:.4f} ({ckpt['iou']*100:.2f}%)")
print(f"  Dice: {ckpt['dice']:.4f}")
print(f"  Loss: {ckpt['loss']:.4f}")
