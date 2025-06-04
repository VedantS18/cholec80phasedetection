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
    def __init__(self, root, split='train', resize=(112, 112), verbose=True):
        """
        Dataset to load full videos from Cholec80 with per-frame phase labels.

        Args:
            root (str): Path to "data/frames"
            split (str): One of ['train', 'val', 'test']
            resize (tuple): Resize frames to this size
            verbose (bool): Debug logging
        """
        self.root = root
        self.resize = resize
        self.samples = []

        all_videos = sorted([v for v in os.listdir(root) if os.path.isdir(os.path.join(root, v))])
        split_indices = {
            'train': all_videos[:60],
            'val': all_videos[60:70],
            'test': all_videos[70:]
        }

        if split not in split_indices:
            raise ValueError(f"Invalid split '{split}', choose from 'train', 'val', 'test'")
        selected_videos = split_indices[split]

        if verbose:
            print("=== Dataset Initialization ===")
            print(f"Root: {self.root}")
            print(f"Split: {split} ({len(selected_videos)} videos)")

        for video in selected_videos:
            video_path = os.path.join(root, video)
            label_file = os.path.join(root.replace("frames", "phase_annotations"), f"{video}-phase.txt")

            if not os.path.exists(label_file):
                if verbose: print(f"[SKIP] Label file not found: {label_file}")
                continue

            # Load labels
            phase_labels = []
            with open(label_file, 'r') as f:
                for line in f.readlines()[1:]:  # skip header
                    parts = line.strip().split('\t')
                    if len(parts) == 2 and parts[1] in PHASE_MAP:
                        phase_labels.append(PHASE_MAP[parts[1]])

            # Load frame paths
            frame_files = []
            for i in range(len(phase_labels)):
                fname = f"{video}_{i + 1:06d}.png"
                fpath = os.path.join(video_path, fname)
                if os.path.exists(fpath):
                    frame_files.append(fpath)
                else:
                    if verbose: print(f"[STOP] Missing frame: {fpath}")
                    break

            if len(frame_files) != len(phase_labels):
                if verbose: print(f"[SKIP] Frame-label mismatch for {video} ({len(frame_files)} vs {len(phase_labels)})")
                continue

            self.samples.append((frame_files, phase_labels))
            if verbose:
                print(f"[OK] {video}: {len(frame_files)} frames")

        if verbose:
            print(f"=== Total valid videos loaded: {len(self.samples)} ===")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, labels = self.samples[idx]
        imgs = []

        for fpath in frame_paths:
            img = cv2.imread(fpath)
            if img is None:
                raise ValueError(f"Could not read image: {fpath}")
            img = cv2.resize(img, self.resize)
            img = img[:, :, ::-1]  # BGR to RGB
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)  # (T, H, W, C)
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # (T, C, H, W)
        imgs = torch.tensor(imgs).float() / 255.0

        labels = torch.tensor(labels).long()  # (T,)
        return imgs.permute(1, 0, 2, 3), labels  # (C, T, H, W), (T,)

