import torch
from torch.utils.data import DataLoader
from dataset.cholec80_dataset import Cholec80ClipDataset
from train_transformer import TransformerPhaseModel
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
CKPT_PATH = "checkpoints/transformer_epoch_10.pt"
DATA_ROOT = "data/frames"
CLIP_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = TransformerPhaseModel().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# --- Load dataset (sequential mode for autoregressive eval) ---
dataset = Cholec80ClipDataset(root=DATA_ROOT, clip_len=CLIP_LEN, sequential=True, return_video_name=True)

# Only keep first 180000 clips
MAX_EVAL_SAMPLES = 180000
dataset = torch.utils.data.Subset(dataset, range(min(MAX_EVAL_SAMPLES, len(dataset))))

loader = DataLoader(dataset, batch_size=1, shuffle=False)

# --- Autoregressive Evaluation Loop ---
predicted_phases_by_video = defaultdict(list)
true_phases_by_video = defaultdict(list)

print("\nStarting autoregressive evaluation...")

with torch.no_grad():
    for idx, (clip, label, video_name) in enumerate(tqdm(loader, desc="Evaluating")):
        clip = clip.to(DEVICE)
        output = model(clip)
        _, predicted = output.max(1)

        predicted_phases_by_video[video_name[0]].append(predicted.item())
        true_phases_by_video[video_name[0]].append(label.item())

# --- Transition Evaluation ---
def get_transitions(phases):
    return [i for i in range(1, len(phases)) if phases[i] != phases[i - 1]]

total_gt = 0
total_pred = 0
matched_pred_indices = set()
matched_gt_indices = set()
timing_errors = []

all_preds = []
all_labels = []

for video_name in predicted_phases_by_video:
    true_phases = true_phases_by_video[video_name]
    predicted_phases = predicted_phases_by_video[video_name]

    all_preds.extend(predicted_phases)
    all_labels.extend(true_phases)

    true_transitions = get_transitions(true_phases)
    pred_transitions = get_transitions(predicted_phases)

    total_gt += len(true_transitions)
    total_pred += len(pred_transitions)

    for gt_idx in true_transitions:
        for delta in range(-3, 4):
            pred_idx = gt_idx + delta
            if 0 <= pred_idx < len(predicted_phases) and pred_idx in pred_transitions and (video_name, pred_idx) not in matched_pred_indices:
                matched_gt_indices.add((video_name, gt_idx))
                matched_pred_indices.add((video_name, pred_idx))
                timing_errors.append(abs(gt_idx - pred_idx))
                break

precision = len(matched_pred_indices) / total_pred if total_pred else 0
recall = len(matched_gt_indices) / total_gt if total_gt else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
mae = np.mean(timing_errors) if timing_errors else 0

# --- Accuracy Evaluation ---
correct = sum(p == t for p, t in zip(all_preds, all_labels))
accuracy = correct / len(all_labels) if all_labels else 0

print("\n--- Autoregressive Transition Evaluation ---")
print(f"True transitions:       {total_gt}")
print(f"Predicted transitions:  {total_pred}")
print(f"Matched transitions:    {len(matched_pred_indices)}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | MAE: {mae:.2f} frames")

print("\n--- Phase Classification Accuracy ---")
print(f"Correct Predictions: {correct}")
print(f"Total Samples:       {len(all_labels)}")
print(f"Accuracy:            {accuracy:.4f}")
