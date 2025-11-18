# dataset_eeg.py
# Robust EEG dataset loader with deterministic, defensive path resolution
# - Solves duplicated-subject-path problems (e.g., subject_01/subject_01/...)
# - Accepts .npy epoch arrays of shapes: (channels, samples), (1, channels, samples),
#   (n_epochs, channels, samples), (samples,)
# - Ensures returned sample shape is (1, channels, samples)
# - Maps string labels to integers and preserves label_map
# - Provides get_dataloader(subject_folder_or_csv, batch_size=..., shuffle=...)

from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    def __init__(self, csv_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if self.df.empty:
            raise ValueError(f"CSV file {csv_path} is empty")

        # determine base_dir
        if base_dir is None:
            base_dir = csv_path.parent
        self.base_dir = Path(base_dir)

        # Normalize column names: prefer 'epoch_path' and 'label' but accept alternatives
        if "epoch_path" not in self.df.columns and "path" in self.df.columns:
            self.df = self.df.rename(columns={"path": "epoch_path"})
        if "epoch_path" not in self.df.columns:
            # assume first column is path
            self.df["epoch_path"] = self.df.iloc[:, 0].astype(str)

        # Label column handling
        if "label" not in self.df.columns:
            if "y" in self.df.columns:
                self.df = self.df.rename(columns={"y": "label"})
            else:
                # if second column present, treat it as label
                if self.df.shape[1] >= 2:
                    possible_label_col = self.df.columns[1]
                    self.df = self.df.rename(columns={possible_label_col: "label"})
                else:
                    raise ValueError("Could not find a label column in CSV. Add a 'label' or 'y' column.")

        # Build label encoding: if labels are numeric keep them, else map strings -> ints
        if pd.api.types.is_integer_dtype(self.df["label"]):
            self.df["label_enc"] = self.df["label"].astype(int)
            self.label_map = None
        else:
            labels = self.df["label"].astype(str).tolist()
            label_map = {}
            next_idx = 0
            encoded = []
            for lab in labels:
                if lab not in label_map:
                    label_map[lab] = next_idx
                    next_idx += 1
                encoded.append(label_map[lab])
            self.df["label_enc"] = encoded
            self.label_map = label_map

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_epoch_path_str(self, s: str) -> str:
        """Normalize path string (fix backslashes, remove leading ./, etc.)."""
        if s is None:
            return s
        s = s.strip()
        # remove leading ./ or .\ if present
        if s.startswith("./") or s.startswith(".\\"):
            s = s[2:]
        # unify separators to posix style for easier processing (Path handles it anyway)
        s = s.replace("\\", "/")
        return s

    def _resolve_epoch_path(self, epoch_path_str: str) -> Path:
        """
        Resolve epoch_path robustly by trying multiple candidate locations.
        Returns the first candidate Path that exists, else raises FileNotFoundError with details.
        """
        epoch_path_str = self._normalize_epoch_path_str(epoch_path_str)
        epoch_path = Path(epoch_path_str)

        candidates = []

        # Candidate 1: base_dir / epoch_path (common case)
        candidates.append(self.base_dir / epoch_path)

        # Candidate 2: if epoch_path starts with the subject folder name and base_dir already = subject folder,
        #                then remove the leading subject component to avoid base_dir/subject/subject/...
        try:
            base_name = self.base_dir.name  # e.g., 'subject_01'
            epoch_parts = epoch_path.parts
            if len(epoch_parts) > 0 and epoch_parts[0] == base_name:
                # take epoch_path without the leading subject part
                new_rel = Path(*epoch_parts[1:]) if len(epoch_parts) > 1 else Path(epoch_path.name)
                candidates.append(self.base_dir / new_rel)
                # also try parent / epoch_path (covers different base_dir choices)
                candidates.append(self.base_dir.parent / epoch_path)
        except Exception:
            # fail-safe: ignore and proceed
            pass

        # Candidate 3: base_dir.parent / epoch_path (if CSV had subject prefixed and base_dir was parent)
        if self.base_dir.parent != self.base_dir:
            candidates.append(self.base_dir.parent / epoch_path)

        # Candidate 4: epoch_path as-is (absolute or relative to cwd)
        candidates.append(epoch_path)

        # Candidate 5: base_dir / epoch_path.name (if CSV used only filenames)
        candidates.append(self.base_dir / epoch_path.name)

        # Candidate 6: try cutting any duplicated subject occurrences like 'subject_01/subject_01/...'
        # Convert to list of strings and attempt to remove repeated blocks
        try:
            parts = list(epoch_path.parts)
            if len(parts) >= 2:
                # remove any repeated consecutive identical parts
                dedup_parts = []
                prev = None
                for p in parts:
                    if p != prev:
                        dedup_parts.append(p)
                    prev = p
                if dedup_parts != parts:
                    candidates.append(self.base_dir / Path(*dedup_parts))
                    candidates.append(self.base_dir.parent / Path(*dedup_parts))
        except Exception:
            pass

        # Deduplicate candidate list while preserving order
        uniq = []
        seen = set()
        for c in candidates:
            try:
                key = str(c.resolve()) if c.exists() else str(c)
            except Exception:
                key = str(c)
            if key not in seen:
                seen.add(key)
                uniq.append(c)

        tried = []
        for cand in uniq:
            tried.append(str(cand))
            if cand.exists():
                return cand.resolve()

        # If none matched, raise a helpful message listing candidates tried
        raise FileNotFoundError(
            "Epoch file not found. Tried the following candidate paths (in order):\n  - "
            + "\n  - ".join(tried)
            + f"\n\n(base_dir was: {self.base_dir})\n(If your CSV stores paths like 'subject_01/...', try passing base_dir as the parent folder 'normalized_epochs' instead of the subject folder.)"
        )

    def __getitem__(self, idx: int):
        """
        Returns:
            x: torch.FloatTensor shaped (1, channels, samples)
            y: torch.LongTensor scalar
        """
        row = self.df.iloc[idx]
        epoch_path_str = str(row["epoch_path"])

        data_path = self._resolve_epoch_path(epoch_path_str)

        arr = np.load(str(data_path))

        # Normalize shapes -> goal is (channels, samples)
        if arr.ndim == 3:
            # (n_epochs, channels, samples) or (1, channels, samples) -> pick first epoch
            arr = arr[0]
        elif arr.ndim == 1:
            # (samples,) -> interpret as single channel
            arr = arr[np.newaxis, :]

        if arr.ndim != 2:
            raise ValueError(f"Unexpected epoch array shape {arr.shape} in file {data_path}")

        # Convert to torch Tensor and add leading channel dim: (1, channels, samples)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

        y_val = row["label_enc"]
        y = torch.tensor(int(y_val), dtype=torch.long)

        return x, y


# Helper to create dataloaders
def get_dataloader(subject_path_or_csv: Union[str, Path], *, batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    p = Path(subject_path_or_csv)

    if p.is_dir():
        # look for common csv names inside the directory
        candidates = ["train_list.csv", "all_train.csv", "train.csv", "all.csv"]
        csv_path = None
        for c in candidates:
            cp = p / c
            if cp.exists():
                csv_path = cp
                break
        if csv_path is None:
            csvs = list(p.glob("*.csv"))
            if csvs:
                csv_path = csvs[0]
            else:
                raise FileNotFoundError(f"No CSV found inside subject folder {p}. Expected one of {candidates} or any .csv.")
        base_dir = p
    else:
        # p is a file (CSV) or path to a CSV
        if p.exists() and p.suffix.lower() == ".csv":
            csv_path = p
            base_dir = p.parent
        else:
            raise FileNotFoundError(f"Path {p} not found or is not a CSV or folder.")

    train_dataset = EEGDataset(csv_path, base_dir=base_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # find validation/test CSV
    val_loader = None
    val_candidates = ["val_list.csv", "test_list.csv", "all_test.csv", "val.csv", "test.csv"]
    for vc in val_candidates:
        vpath = base_dir / vc
        if vpath.exists():
            val_ds = EEGDataset(vpath, base_dir=base_dir)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            break

    return train_loader, val_loader


# Quick test when run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        candidate = sys.argv[1]
    else:
        candidate = "normalized_epochs/subject_01"

    print("Testing dataloader for:", candidate)
    train_loader, val_loader = get_dataloader(candidate, batch_size=1)
    it = iter(train_loader)
    x, y = next(it)
    print("Batch x shape:", x.shape)   # expected: (1, 1, C, T)
    print("Batch y shape:", y.shape, "value:", y)
    ds = train_loader.dataset
    print("Label mapping (label_map):", getattr(ds, "label_map", None))
