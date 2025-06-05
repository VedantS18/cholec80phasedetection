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
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------- Config ----------------
DATA_ROOT = "data/frames"
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_hmm(emissions, labels, num_classes=7):
    model = hmm.MultinomialHMM(n_components=num_classes, n_iter=100, tol=1e-4)
    flat_emissions = torch.cat(emissions).argmax(dim=1).unsqueeze(1).numpy()
    flat_labels = torch.cat(labels).numpy()
    lengths = [e.size(0) for e in emissions]
    model.fit(flat_emissions, lengths)
    return model

def decode_with_hmm(model, emissions):
    pred_seq = emissions.argmax(dim=1).unsqueeze(1).numpy()
    viterbi_seq = model.predict(pred_seq)
    return torch.tensor(viterbi_seq)

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
            x = x.view(x.size(0), -1)
        return x

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

def create_multi_video_loader(dataset, indices, batch_size):
    paired_indices = [(indices[i], indices[i+1]) for i in range(0, len(indices) - 1, 2)]
    def collate_fn(pairs):
        batch_inputs, batch_labels = [], []
        for idx1, idx2 in pairs:
            vid1 = dataset[idx1]
            vid2 = dataset[idx2]
            input = torch.cat([vid1[0], vid2[0]], dim=1)
            label = torch.cat([vid1[1], vid2[1]], dim=0)
            batch_inputs.append(input)
            batch_labels.append(label)
        return batch_inputs, batch_labels
    return DataLoader(paired_indices, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    full_dataset = Cholec80FullVideoDataset(DATA_ROOT)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = indices[:50], indices[50:65]

    train_loader = create_multi_video_loader(full_dataset, train_idx, BATCH_SIZE)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False)

    feature_extractor = ResNetFeatureExtractor().to(device)
    decoder = CausalTransformer().to(device)
    decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        decoder.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} Rank {rank}"):
            inputs, labels = inputs[0].to(device), labels[0].to(device)
            features = [feature_extractor(inputs[:, t].unsqueeze(0)).squeeze(0) for t in range(inputs.shape[1])]
            features = torch.stack(features, dim=0)
            outputs = decoder(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        decoder.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.squeeze(0).to(device), labels.squeeze(0).to(device)
                features = [feature_extractor(clips[:, t].unsqueeze(0)).squeeze(0) for t in range(clips.shape[1])]
                features = torch.stack(features, dim=0)
                outputs = decoder(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.append(torch.argmax(outputs, dim=1))
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)
        avg_val_loss = val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(decoder.module.state_dict(), "best_model.pt")
                print("[Saved new best model]")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
