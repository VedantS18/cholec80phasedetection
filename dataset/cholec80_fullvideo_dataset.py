import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

PHASE_MAP = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderRetraction": 4,
    "CleaningCoagulation": 5,
    "GallbladderPackaging": 6
}

class Cholec80FullVideoDataset(Dataset):
    def __init__(self, root, resize=(112, 112), video_list=None, verbose=False):
        self.root = root
        self.resize = resize
        self.samples = []

        print("=== Dataset Initialization ===")
        print(f"Root: {self.root}")

        all_videos = sorted(os.listdir(root))
        selected_videos = video_list if video_list is not None else all_videos

        print(f"Found {len(all_videos)} videos, loading {len(selected_videos)}")

        for video in selected_videos:
            video_path = os.path.join(root, video)
            if not os.path.isdir(video_path):
                continue

            label_file = os.path.join(root.replace("frames", "phase_annotations"), f"{video}-phase.txt")
            if not os.path.exists(label_file):
                if verbose: print(f"[SKIP] Label file not found for {video}")
                continue

            # Load every 25th label
            with open(label_file, 'r') as f:
                lines = f.readlines()[1:]  # skip header
                all_labels = []
                for i in range(0, len(lines), 25):
                    parts = lines[i].strip().split('\t')
                    if len(parts) == 2 and parts[1] in PHASE_MAP:
                        all_labels.append(PHASE_MAP[parts[1]])

            # Load corresponding frame paths
            frame_files = []
            for i in range(len(all_labels)):
                fname = f"{video}_{i+1:06d}.png"
                fpath = os.path.join(video_path, fname)
                if os.path.exists(fpath):
                    frame_files.append(fpath)
                else:
                    if verbose: print(f"[STOP] Missing frame: {fpath}")
                    break

            if len(frame_files) != len(all_labels):
                if verbose: print(f"[TRIM] Truncating labels from {len(all_labels)} to {len(frame_files)}")
                all_labels = all_labels[:len(frame_files)]

            if len(frame_files) == len(all_labels) and len(frame_files) > 0:
                self.samples.append((frame_files, all_labels))
                if verbose: print(f"[OK] Loaded {video}: {len(frame_files)} frames")
            else:
                if verbose: print(f"[SKIP] Frame-label mismatch for {video} ({len(frame_files)} frames vs {len(all_labels)} labels)")

        print(f"=== Total valid videos loaded: {len(self.samples)} ===")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, labels = self.samples[idx]
        imgs = []

        for fpath in frame_paths:
            img = cv2.imread(fpath)
            img = cv2.resize(img, self.resize)
            img = img[:, :, ::-1]  # BGR to RGB
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)  # (T, H, W, C)
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # (T, C, H, W)
        imgs = torch.tensor(imgs).float() / 255.0  # Normalize

        labels = torch.tensor(labels).long()  # (T,)
        return imgs.permute(1, 0, 2, 3), labels  # (C, T, H, W), (T,)
