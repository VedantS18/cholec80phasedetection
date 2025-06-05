import os
import random
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
import numpy as np
from dataset.cholec80_fullvideo_dataset import Cholec80FullVideoDataset
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

# ---------------- Config ----------------
DATA_ROOT = "data/frames"
MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_FRAMES = False  # Set to True if you want to save image files
SAVE_DIR = "correct_frames"

# ---------------- Model ----------------
def get_sinusoid_encoding(n_position, d_hid):
    position = torch.arange(n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float) * -(np.log(10000.0) / d_hid))
    pe = torch.zeros(n_position, d_hid)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

class CausalTransformer(nn.Module):
    def __init__(self, input_dim=512, num_classes=7, nhead=4, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        T = x.size(0)
        pos_embed = get_sinusoid_encoding(T, x.size(-1)).to(x.device)
        x = x + pos_embed
        x = x.unsqueeze(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.transformer(x, mask=mask)
        return self.classifier(out.squeeze(1))

# ---------------- Evaluation ----------------
def visualize_correct():
    dataset = Cholec80FullVideoDataset(DATA_ROOT)
    test_idx = list(range(70, 75))
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)

    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    model = CausalTransformer().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.eval()

    if SAVE_FRAMES and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    transform = T.ToPILImage()
    correct = []

    with torch.no_grad():
        for video_id, (clips, labels) in enumerate(test_loader, start=65):
            clips = clips.squeeze(0).to(DEVICE)
            labels = labels.squeeze(0).to(DEVICE)

            features = []
            for t in range(clips.shape[1]):
                frame = clips[:, t]
                ft = feature_extractor(frame.unsqueeze(0))
                features.append(ft.squeeze(0))
            features = torch.stack(features, dim=0)

            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(preds)):
                if preds[i].item() == labels[i].item():
                    correct.append((video_id, i, preds[i].item(), clips[:, i].cpu()))

    print(f"Total correct frames: {len(correct)}")
    sample = random.sample(correct, min(20, len(correct)))
    for vid, frame, phase, img_tensor in sample:
        print(f"Video {vid:02d} - Frame {frame:04d}: Phase {phase}")
        if SAVE_FRAMES:
            img = transform(img_tensor)
            img.save(os.path.join(SAVE_DIR, f"correct_vid{vid:02d}_frame{frame:04d}.png"))

if __name__ == "__main__":
    visualize_correct()
