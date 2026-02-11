import torch

print("GPU Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("PyTorch:", torch.__version__)
print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB" if torch.cuda.is_available() else "N/A")
