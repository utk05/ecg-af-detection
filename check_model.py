import torch
from ecg_model import load_model

model = load_model('model_v4.pt')
print("Params:", sum(p.numel() for p in model.parameters()))
with torch.no_grad():
    out = model(torch.randn(2, 1, 9000))
print("Output shape:", tuple(out.shape))
print("Probabilities sum to:", torch.softmax(out, dim=1).sum(dim=1))