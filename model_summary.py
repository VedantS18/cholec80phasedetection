# model_summary.py

import torch
from train_transformer import ResNetFeatureExtractor, CausalTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    decoder = CausalTransformer().to(DEVICE)

    print("=== Model Parameter Counts ===")
    print(f"Feature Extractor (ResNet18): {count_parameters(feature_extractor):,} trainable parameters")
    print(f"Transformer Decoder: {count_parameters(decoder):,} trainable parameters")
    print(f"Total: {count_parameters(feature_extractor) + count_parameters(decoder):,} trainable parameters")

if __name__ == "__main__":
    main()
