import os
import torch
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset

class Cholec80ClipDataset(Dataset):
    def __init__(self, frames_dir, labels_dir, clip_len=16, resize=(112, 112), transform=None):
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.clip_len = clip_len
        self.resize = resize
        self.transform = transform

        self.video_dirs = sorted(glob.glob(os.path.join(frames_dir, "*")))
        self.samples = []

        for vid_idx, video_path in enumerate(self.video_dirs):
            label_path = os.path.join(labels_dir, os.path.basename(video_path) + ".txt")
            with open(label_path, 'r') as f:
                labels = [int(line.strip()) for line in f]

            num_clips = len(labels) - clip_len
            for i in range(num_clips):
                self.samples.append((video_path, label_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_path, start_idx = self.samples[idx]
        frames = []

        for i in range(start_idx, start_idx + self.clip_len):
            frame_path = os.path.join(video_path, f"frame_{i:05d}.jpg")
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, self.resize)
            frame = frame[:, :, ::-1]  # BGR to RGB
            frames.append(frame)

        clip = np.stack(frames, axis=0)  # (T, H, W, C)
        clip = clip.transpose(3, 0, 1, 2)  # (C, T, H, W)
        clip = torch.tensor(clip, dtype=torch.float32) / 255.0

        label_path = os.path.join(label_path)
        with open(label_path, 'r') as f:
            labels = [int(line.strip()) for line in f]
        label = labels[start_idx + self.clip_len // 2]  # Middle frame label

        return clip, label
