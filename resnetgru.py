import torch.nn as nn

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.size(0), -1)
        return x

class PhaseGRU(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=7):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: (B, T, D)
        out, _ = self.gru(x)
        return self.classifier(out)

