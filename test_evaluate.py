import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from dataset.cholec80_fullvideo_dataset import Cholec80FullVideoDataset

# ---------------- Config ----------------
DATA_ROOT = "data/frames"
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pt"

# ---------------- Helper ----------------
def get_sinusoid_encoding(n_position, d_hid):
    position = torch.arange(n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float) * -(np.log(10000.0) / d_hid))
    pe = torch.zeros(n_position, d_hid)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def compute_metrics(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    total = len(labels)
    correct = (preds == labels).sum()
    accuracy = correct / total

    transitions_true = np.where(labels[1:] != labels[:-1])[0] + 1
    transitions_pred = np.where(preds[1:] != preds[:-1])[0] + 1
    matched = sum(t in transitions_pred for t in transitions_true)
    precision = matched / len(transitions_pred) if transitions_pred.size else 0
    recall = matched / len(transitions_true) if transitions_true.size else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return accuracy, precision, recall, f1

# ---------------- Models ----------------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.size(0), -1)
        return x

class CausalTransformer(nn.Module):
    def __init__(self, input_dim=512, num_classes=7, nhead=4, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x: (T, D)
        T = x.size(0)
        pos_embed = get_sinusoid_encoding(T, x.size(-1)).to(x.device)
        x = x + pos_embed
        x = x.unsqueeze(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.transformer(x, mask=mask)
        return self.classifier(out.squeeze(1))

# ---------------- Evaluation ----------------
def evaluate():
    dataset = Cholec80FullVideoDataset(DATA_ROOT)
    indices = list(range(len(dataset)))
    test_idx = indices[72:75]  # use same split as training
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    model = CausalTransformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    feature_extractor.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0
    misclassified = []  # list of (video_idx, frame_idx, pred, label)

    with torch.no_grad():
        for i, (clips, labels) in enumerate(test_loader):
            video_idx = test_idx[i]
            clips, labels = clips.squeeze(0).to(DEVICE), labels.squeeze(0).to(DEVICE)
            features = []
            for t in range(clips.shape[1]):
                frame = clips[:, t]
                ft = feature_extractor(frame.unsqueeze(0))
                features.append(ft.squeeze(0))
            features = torch.stack(features, dim=0)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred_classes = torch.argmax(outputs, dim=1)
            all_preds.append(pred_classes)
            all_labels.append(labels)

            for idx, (pred, gt) in enumerate(zip(pred_classes, labels)):
                if pred.item() != gt.item():
                    misclassified.append((video_idx, idx, pred.item(), gt.item()))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)

    print("\n=== Test Set Evaluation ===")
    print(f"Loss: {total_loss / len(test_loader):.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nTotal Misclassified Frames: {len(misclassified)}")

    import random

    sample_size = min(20, len(misclassified))
    random_samples = random.sample(misclassified, sample_size)

    print(f"\nShowing {sample_size} randomly sampled misclassified frames:")
    for vid, frame, pred, gt in random_samples:
        print(f"Video {vid:02d} - Frame {frame:04d}: Predicted {pred}, Ground Truth {gt}")

if __name__ == "__main__":
    evaluate()
