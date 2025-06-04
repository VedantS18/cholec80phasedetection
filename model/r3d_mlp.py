import torch.nn as nn
import torchvision

class R3DMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.video.r3d_18(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)  # B x 512
        return self.mlp_head(features)
