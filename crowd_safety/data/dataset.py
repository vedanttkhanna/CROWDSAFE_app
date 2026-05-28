
"""
Crowd Trajectory Dataset
Parses ETH/UCY obsmat format into (obs, pred) window pairs.

obsmat columns: frame_id, ped_id, x, z, y, vx, vz, vy
We use x, y (columns 2 and 4) as 2D world coordinates.

Standard benchmark split:
  obs_len  = 8  frames (3.2 seconds)
  pred_len = 12 frames (4.8 seconds)
  step     = 1  frame  (0.4 seconds per frame)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


OBS_LEN  = 8
PRED_LEN = 12
SKIP     = 1

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def load_obsmat(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 5:
                continue
            frame = int(float(vals[0]))
            ped   = int(float(vals[1]))
            x     = float(vals[2])
            y     = float(vals[4])
            data.setdefault(ped, []).append((frame, x, y))
    for ped in data:
        data[ped].sort(key=lambda t: t[0])
    return data


def extract_windows(ped_data, obs_len=OBS_LEN, pred_len=PRED_LEN, skip=SKIP):
    total_len = obs_len + pred_len
    windows   = []
    ped_ids   = []
    for ped_id, traj in ped_data.items():
        frames = np.array([t[0] for t in traj])
        coords = np.array([[t[1], t[2]] for t in traj])
        for start in range(0, len(traj) - total_len + 1, skip):
            end = start + total_len
            frame_slice = frames[start:end]
            diffs = np.diff(frame_slice)
            if not np.all(diffs == diffs[0]):
                continue
            obs  = coords[start:start + obs_len]
            pred = coords[start + obs_len:end]
            windows.append((obs, pred))
            ped_ids.append(ped_id)
    return windows, ped_ids


def normalize_windows(windows):
    norm_windows = []
    origins      = []
    for obs, pred in windows:
        origin = obs[0].copy()
        obs_n  = obs  - origin
        pred_n = pred - origin
        norm_windows.append((obs_n, pred_n))
        origins.append(origin)
    return norm_windows, np.array(origins)


class TrajectoryDataset(Dataset):
    def __init__(self, scenes, data_dir, obs_len=OBS_LEN, pred_len=PRED_LEN):
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.obs_seqs  = []
        self.pred_seqs = []
        self.ped_ids   = []
        self.origins   = []
        self.scene_labels = []

        for scene in scenes:
            path = os.path.join(data_dir, scene, "obsmat.txt")
            if not os.path.exists(path):
                print(f"  [WARN] missing {path}, skipping")
                continue
            ped_data = load_obsmat(path)
            windows, pids = extract_windows(ped_data, obs_len, pred_len)
            if len(windows) == 0:
                print(f"  [WARN] no windows extracted from {scene}")
                continue
            norm_windows, origins = normalize_windows(windows)
            for (obs_n, pred_n), origin, pid in zip(norm_windows, origins, pids):
                self.obs_seqs.append(obs_n)
                self.pred_seqs.append(pred_n)
                self.origins.append(origin)
                self.ped_ids.append(pid)
                self.scene_labels.append(scene)

        self.obs_seqs  = np.array(self.obs_seqs,  dtype=np.float32)
        self.pred_seqs = np.array(self.pred_seqs, dtype=np.float32)
        self.origins   = np.array(self.origins,   dtype=np.float32)
        print(f"  Loaded {len(self.obs_seqs)} windows from {scenes}")

    def __len__(self):
        return len(self.obs_seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs_seqs[idx]),
            torch.tensor(self.pred_seqs[idx]),
        )


def get_loaders(data_dir, test_scene="eth", batch_size=64):
    train_scenes = [s for s in SCENES if s != test_scene]
    test_scenes  = [test_scene]
    train_ds_full = TrajectoryDataset(train_scenes, data_dir)
    test_ds       = TrajectoryDataset(test_scenes,  data_dir)
    n_val   = int(0.15 * len(train_ds_full))
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
