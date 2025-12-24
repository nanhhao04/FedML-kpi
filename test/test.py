import torch
from torch import nn
from src.models import VAE

# Khởi tạo mô hình
model = VAE(input_dim=15, hidden_dim=64, latent_dim=16)

# Đếm số tham số
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

# In chi tiết từng layer
print("\n=== Layer-wise parameter count ===")
for name, p in model.named_parameters():
    print(f"{name:20s} : {p.numel()} params")
