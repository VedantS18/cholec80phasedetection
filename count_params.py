import torch
from torch import nn
from phase_gru import PhaseGRU, ResNetFeatureExtractor

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    # Instantiate models
    feature_extractor = ResNetFeatureExtractor()
    gru_model = PhaseGRU()

    # Count parameters
    total_feat, train_feat = count_parameters(feature_extractor)
    total_gru, train_gru = count_parameters(gru_model)

    print("=== Parameter Counts ===")
    print(f"Feature Extractor - Total: {total_feat:,}, Trainable: {train_feat:,}")
    print(f"PhaseGRU Model    - Total: {total_gru:,}, Trainable: {train_gru:,}")
    print(f"Combined           - Total: {total_feat + total_gru:,}, Trainable: {train_feat + train_gru:,}")
