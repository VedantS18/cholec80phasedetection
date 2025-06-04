import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import numpy as np
from dataset.cholec80_fullvideo_dataset import Cholec80FullVideoDataset
from hmmlearn import hmm

# ---------------- Config ----------------
DATA_ROOT = "data/frames"
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ---------------- HMM ----------------
def train_hmm(emissions, labels, num_classes=7):
    model = hmm.GaussianHMM(
        n_components=num_classes,
        covariance_type="diag",
        n_iter=100,
        tol=1e-4,
        init_params="stmc",
        params="stmc",
    )
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        probs = [softmax(e + 1e-6) for e in emissions]
        flat_emissions = torch.cat(probs, dim=0).numpy()
        lengths = [e.shape[0] for e in emissions]
    model.fit(flat_emissions, lengths)
    return model

def decode_with_hmm(model, emissions):
    softmax = nn.Softmax(dim=1)
    probs = softmax(emissions + 1e-6)
    viterbi_seq = model.predict(probs.numpy())
    return torch.tensor(viterbi_seq)

# ---------------- Helper ----------------
def get_sinusoid_encoding(n_position, d_hid):
    position = torch.arange(n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float) * -(np.log(10000.0) / d_hid))
    pe = torch.zeros(n_position, d_hid)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (n_position, d_hid)

# ---------------- Model ----------------
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
        x = x.unsqueeze(1)  # (T, 1, D)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()  # causal mask
        out = self.transformer(x, mask=mask)
        return self.classifier(out.squeeze(1))  # (T, num_classes)

# ---------------- Metrics ----------------
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

# ---------------- Train ----------------
def train():
    full_dataset = Cholec80FullVideoDataset(DATA_ROOT)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx, test_idx = indices[:50], indices[50:65], indices[65:]

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False)

    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    decoder = CausalTransformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    print("=== Starting Training ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        decoder.train()
        train_loss = 0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            clips, labels = clips.squeeze(0).to(DEVICE), labels.squeeze(0).to(DEVICE)
            features = []
            for t in range(clips.shape[1]):
                frame = clips[:, t]
                ft = feature_extractor(frame.unsqueeze(0))
                features.append(ft.squeeze(0))
            features = torch.stack(features, dim=0)
            outputs = decoder(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        decoder.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.squeeze(0).to(DEVICE), labels.squeeze(0).to(DEVICE)
                features = []
                for t in range(clips.shape[1]):
                    frame = clips[:, t]
                    ft = feature_extractor(frame.unsqueeze(0))
                    features.append(ft.squeeze(0))
                features = torch.stack(features, dim=0)
                outputs = decoder(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        new_preds, new_labels = [], []
        hmm_model = train_hmm(all_preds, all_labels)
        for emissions, labels in zip(all_preds, all_labels):
            viterbi_preds = decode_with_hmm(hmm_model, emissions)
            new_preds.append(viterbi_preds)
            new_labels.append(labels)
        new_preds = torch.cat(new_preds)
        new_labels = torch.cat(new_labels)
        acc, prec, rec, f1 = compute_metrics(new_preds, new_labels)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), "best_model.pt")
            print("[Saved new best model]")

if __name__ == "__main__":
    train()
