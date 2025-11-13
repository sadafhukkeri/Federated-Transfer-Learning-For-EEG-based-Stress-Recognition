"""
preprocess_sam40.py

Usage:
    python preprocess_sam40.py --raw_dir path/to/raw_dataset --out_dir path/to/output --fs 128

What it does:
- Scans raw_dir for files (supports .npy, .csv, .mat)
- Infers subject id and task from filename (see TASK_KEYWORDS)
- Splits each recording into non-overlapping epochs (epoch_len_s)
- Normalizes each epoch per-channel (z-score)
- Saves epochs as .npy under out_dir/subject_x/{stress,relaxed}/epoch_xxx.npy
- Writes metadata.csv per subject and train/val lists
"""

import os
import re
import argparse
import numpy as np
import math
import csv
from pathlib import Path

# Optional: for .mat files
try:
    import scipy.io as sio
except Exception:
    sio = None

# ---- Configurable keywords (adjust if needed) ----
TASK_KEYWORDS = {
    "stress": ["stroop", "arithmetic", "mirror"],
    "relaxed": ["relax", "baseline", "rest"]
}

# default epoch settings for SAM40
EPOCH_LEN_S = 25  # dataset epochs are 25s, but we also support splitting longer files
DEFAULT_FS = 128  # UPDATE to actual sampling rate from dataset

# file extensions we support
SUPPORTED_EXT = [".npy", ".csv", ".mat"]


# ---- Helpers ----
def infer_label_from_filename(fname: str):
    """Return 'stress' or 'relaxed' (or None) based on keywords in filename."""
    fn = fname.lower()
    for label, keys in TASK_KEYWORDS.items():
        for k in keys:
            if k in fn:
                return label
    return None


def load_signal(filepath: Path):
    """Load data from .npy, .csv, or .mat. Return numpy array (2D)."""
    ext = filepath.suffix.lower()
    if ext == ".npy":
        return np.load(filepath)
    elif ext == ".csv":
        # Try numpy first for fast load (works if there is no header and all numeric)
        try:
            arr = np.loadtxt(filepath, delimiter=",")
            if arr.size == 0:
                raise ValueError("Empty CSV")
            # Ensure 2D
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return arr
        except Exception:
            # pandas load & cleaning
            try:
                import pandas as pd
            except Exception as e:
                raise RuntimeError("pandas is required to robustly load CSV files. Install pandas.") from e

            # Read CSV, let pandas infer header; this handles files with headers or mixed types
            df = pd.read_csv(filepath, header=0, dtype=object)

            # Convert all columns to numeric where possible; non-convertible become NaN
            df = df.apply(pd.to_numeric, errors="coerce")

            # Drop columns and rows that are entirely NaN
            df.dropna(axis=0, how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)

            if df.empty:
                raise ValueError(f"No numeric data found in CSV: {filepath}")

            # Heuristic: if first column looks like a time/index column (monotonically non-decreasing and many uniques),
            # drop it (common case where CSV includes a time column)
            try:
                if df.shape[1] > 32:
                    first_col = df.iloc[:, 0].values
                    # monotonic non-decreasing check + many unique values indicates time/index column
                    if (np.all(np.diff(first_col) >= 0) and (len(np.unique(first_col)) > df.shape[0] * 0.5)):
                        df = df.iloc[:, 1:]
            except Exception:
                # If any error occurs in heuristic, ignore and keep all columns
                pass

            arr = df.values
            # If arr is shape (n_time, n_channels) but we expect channels x time, we'll handle later in ensure_channels_first
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return arr.astype(np.float32)
    elif ext == ".mat":
        if sio is None:
            raise RuntimeError("scipy is required to load .mat files (scipy.io).")
        mat = sio.loadmat(filepath)
        # Heuristic: pick the first 2D numeric matrix in mat
        for k, v in mat.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                return v
        raise ValueError(f"No 2D array found in {filepath}")
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def ensure_channels_first(arr: np.ndarray):
    """
    Convert array to shape (n_channels, n_time).
    Accepts (n_time, n_channels) or (n_channels, n_time).
    """
    if arr.ndim != 2:
        raise ValueError("Expected 2D array [channels x time] or [time x channels].")
    # If first dim is small (<=64) and second is large, assume channels x time
    if arr.shape[0] <= 64 and arr.shape[1] > arr.shape[0]:
        return arr
    # If second dim is small (<=64) and first is large, assume time x channels -> transpose
    if arr.shape[1] <= 64 and arr.shape[0] > arr.shape[1]:
        return arr.T
    # If both dims are similar, try to guess: if any dimension equals 32 (expected channels), prefer that
    if 32 in arr.shape:
        if arr.shape[0] == 32:
            return arr
        else:
            return arr.T
    # Fallback: assume channels are the smaller dimension
    if arr.shape[0] < arr.shape[1]:
        return arr
    else:
        return arr.T


def zscore_epoch(epoch: np.ndarray, eps: float = 1e-8):
    """Z-score per channel: (x-mean)/std"""
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True) + eps
    return (epoch - mean) / std


def split_into_epochs(data: np.ndarray, fs: int, epoch_len_s: int):
    """Split channels x time array into non-overlapping epochs of epoch_len_s seconds."""
    n_samples = data.shape[1]
    samples_per_epoch = int(fs * epoch_len_s)
    n_epochs = n_samples // samples_per_epoch
    epochs = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epochs.append(data[:, start:end])
    return epochs


# ---- Main processing function ----
def process_dataset(raw_dir: str, out_dir: str, fs: int = DEFAULT_FS, epoch_len_s: int = EPOCH_LEN_S, overwrite=False):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    for ext in SUPPORTED_EXT:
        all_files.extend(raw_dir.rglob(f"*{ext}"))

    if not all_files:
        raise FileNotFoundError(f"No files with extensions {SUPPORTED_EXT} found under {raw_dir}")

    # Group files by inferred subject ID (heuristic: look for subject* or s* in filename or parent folder)
    # If filenames contain "sub01" or "subject_01" or "S1", we'll try to extract numeric id.group by parent folder name.
    subject_pattern = re.compile(r"(sub|subject|s|subj)[-_]?0*([0-9]{1,3})", re.IGNORECASE)

    grouped = {}
    for f in all_files:
        name = f.name
        # try pattern in filename
        m = subject_pattern.search(name)
        if m:
            sid = int(m.group(2))
            key = f"subject_{sid:02d}"
        else:
            # fallback: parent folder name (use last part)
            parent = f.parent.name
            # sanitize
            key = re.sub(r"[^\w\d_]", "_", parent).lower()
        grouped.setdefault(key, []).append(f)

    print(f"Found {len(grouped)} subjects (groups) in the raw folder.")

    # Process each subject group
    for subj_key, files in sorted(grouped.items()):
        subj_out = out_dir / subj_key
        stress_dir = subj_out / "stress"
        relax_dir = subj_out / "relaxed"
        stress_dir.mkdir(parents=True, exist_ok=True)
        relax_dir.mkdir(parents=True, exist_ok=True)

        metadata = []  # list of (epoch_path, label, src_file)
        print(f"\nProcessing {subj_key} with {len(files)} files...")

        epoch_counter = {"stress": 0, "relaxed": 0}
        for fpath in sorted(files):
            label = infer_label_from_filename(fpath.name)
            if label is None:
                # attempt to infer from parent folder
                label = infer_label_from_filename(fpath.parent.name)
            if label is None:
                print(f"  WARNING: Could not infer label for {fpath.name}. Skipping.")
                continue

            try:
                arr = load_signal(fpath)
            except Exception as e:
                print(f"  ERROR loading {fpath}: {e}. Skipping.")
                continue

            try:
                arr = ensure_channels_first(arr)
            except Exception as e:
                print(f"  ERROR shape for {fpath}: {e}. Skipping.")
                continue

            # split into 25s epochs (or epoch_len_s)
            epochs = split_into_epochs(arr, fs=fs, epoch_len_s=epoch_len_s)
            if not epochs:
                print(f"  WARNING: No full {epoch_len_s}s epochs in {fpath.name} (len={arr.shape[1]} samples). Skipping.")
                continue

            for ep in epochs:
                # normalize
                ep_norm = zscore_epoch(ep)
                # save
                idx = epoch_counter[label]
                fname_out = f"epoch_{idx:04d}.npy"
                if label == "stress":
                    out_path = stress_dir / fname_out
                else:
                    out_path = relax_dir / fname_out

                if out_path.exists() and not overwrite:
                    # skip saving duplicate
                    pass
                else:
                    np.save(out_path, ep_norm.astype(np.float32))
                metadata.append((str(out_path.relative_to(out_dir)), label, str(fpath.name)))
                epoch_counter[label] += 1

        # write metadata.csv
        meta_csv = subj_out / "metadata.csv"
        with open(meta_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch_path", "label", "src_file"])
            for row in metadata:
                writer.writerow(row)

        # train/val split lists (80/20) based on per-subject epochs
        # read all saved epochs
        saved_epochs = []
        for lab in ("stress", "relaxed"):
            p = subj_out / lab
            if not p.exists():
                continue
            for e in sorted(p.glob("epoch_*.npy")):
                saved_epochs.append((str(e.relative_to(out_dir)), lab))
        # shuffle deterministically
        saved_epochs.sort()
        n = len(saved_epochs)
        n_train = int(0.8 * n)
        train = saved_epochs[:n_train]
        val = saved_epochs[n_train:]

        # write to files
        with open(subj_out / "train_list.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch_path", "label"])
            writer.writerows(train)
        with open(subj_out / "val_list.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch_path", "label"])
            writer.writerows(val)

        print(f"  Saved {epoch_counter['stress']} stress epochs and {epoch_counter['relaxed']} relaxed epochs for {subj_key}.")
        print(f"  Train/Val: {len(train)}/{len(val)}")

    print("\nProcessing complete. Output saved to:", out_dir)


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw dataset folder")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to output folder for organized epochs")
    parser.add_argument("--fs", type=int, default=DEFAULT_FS, help="Sampling frequency (Hz)")
    parser.add_argument("--epoch_len_s", type=int, default=EPOCH_LEN_S, help="Epoch length in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing epoch files")
    args = parser.parse_args()

    process_dataset(args.raw_dir, args.out_dir, fs=args.fs, epoch_len_s=args.epoch_len_s, overwrite=args.overwrite)
