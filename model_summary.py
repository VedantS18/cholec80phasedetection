import torch
import torch.nn as nn
from resnetgru import PhaseGRU  # adjust the import based on your directory structure

# Load your trained model
MODEL_PATH = "checkpoints/Epoch_10.pt"
model = PhaseGRU(input_dim=512, hidden_dim=256, num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

# Count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

