from models import AELSTM
import torch

model = AELSTM(input_dim=15, enc_hidden=64, dec_hidden=64, bottleneck=32)
x = torch.randn(8, 10, 15)
out = model(x)

print("x shape:   ", x.shape)
print("out shape: ", out.shape)
