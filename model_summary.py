import torch

# Load full model from checkpoint
checkpoint = torch.load("checkpoints/epoch_10.pt", map_location="cpu")

# If the checkpoint is a full model (not just state_dict), skip loading into nn.Module
# Count parameters directly
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

total_params = sum(v.numel() for k, v in state_dict.items())
print(f"Total parameters (from state_dict): {total_params:,}")
