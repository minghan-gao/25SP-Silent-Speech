import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class EchoNPYDataset(Dataset):
    def __init__(self, npy_path, label_path, transform=None):
        # 1) Load labels
        df        = pd.read_csv(label_path, header=None)
        labels    = df[3].tolist()
        n_labels  = len(labels)

        # 2) Load & trim data
        all_data = np.load(npy_path).astype(np.float32)
        if all_data.shape[0] < n_labels:
            raise ValueError(f".npy has only {all_data.shape[0]} rows but you have {n_labels} labels")
        data = all_data[:n_labels]

        # 3) Normalize (optional but recommended)
        mean = data.mean()
        std  = data.std() + 1e-6
        data = (data - mean) / std

        # 4) Store
        self.data      = data
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])       # (L,)
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]                       # always in‐range now
        return x, y