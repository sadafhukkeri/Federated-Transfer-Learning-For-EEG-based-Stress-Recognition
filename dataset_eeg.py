import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from pathlib import Path

class EEGDataset(Dataset):
    def __init__(self, csv_path, base_dir):
        self.df = pd.read_csv(csv_path)
        self.base_dir = Path(base_dir)
        self.label_map = {"relaxed": 0, "stress": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data_path = (self.base_dir / row["epoch_path"]).resolve()
        label = 0 if row["label"] == "relaxed" else 1

        data = np.load(data_path)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)


        return data, label

    def __len__(self):
        return len(self.df)
    
def get_dataloader(subject_path, batch_size=8):
    """Returns train and val dataloaders for a subject"""
    subject_dir = Path(subject_path)
    train_csv = subject_dir / 'train_list.csv'
    val_csv = subject_dir / 'val_list.csv'

    train_dataset = EEGDataset(train_csv, base_dir=subject_dir.parent)
    val_dataset = EEGDataset(val_csv, base_dir=subject_dir.parent)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader