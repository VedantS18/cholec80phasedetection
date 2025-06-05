import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from dataset.cholec80_fullvideo_dataset import Cholec80FullVideoDataset
from phase_gru import ResNetFeatureExtractor, PhaseGRU  # Make sure your model is saved here

# ---------------- Config ----------------
DATA_ROOT = "data/frames"
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_idx, val_idx = indices[:50], indices[50:65]

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False)

    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    model = PhaseGRU().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    print("=== Starting Training ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            clips, labels = clips.squeeze(0).to(DEVICE), labels.squeeze(0).to(DEVICE)  # (C, T, H, W), (T,)
            features = []
            for t in range(clips.shape[1]):
                frame = clips[:, t]
                ft = feature_extractor(frame.unsqueeze(0))  # (1, D)
                features.append(ft.squeeze(0))
            features = torch.stack(features).unsqueeze(0)  # (1, T, D)

            outputs = model(features).squeeze(0)  # (T, num_classes)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
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
                features = torch.stack(features).unsqueeze(0)

                outputs = model(features).squeeze(0)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_preds.append(torch.argmax(outputs, dim=1))
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_gru_model.pt")
            print("[Saved new best model]")

if __name__ == "__main__":
    train()
