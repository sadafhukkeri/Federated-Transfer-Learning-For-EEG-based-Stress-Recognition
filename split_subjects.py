# split_subjects.py
import os
import csv
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def gather_epochs(subject_dir: Path):
    """Return list of tuples (relative_path, label) for all .npy epochs in a subject folder"""
    recs = []
    for state in ("stress", "relaxed"):
        state_dir = subject_dir / state
        if not state_dir.exists():
            continue
        for p in sorted(state_dir.glob("epoch_*.npy")):
            # path relative to data_dir (we will compute that later)
            recs.append((p, state))
    return recs

def write_list_csv(rows, out_path: Path, data_dir: Path):
    """rows: list of (Path, label). Write epoch_path (relative to data_dir), label"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_path", "label"])
        for p, label in rows:
            rel = str(p.relative_to(data_dir)).replace("/", os.sep)  # keep os-specific separators
            writer.writerow([rel, label])

def split_subjects(data_dir: str, out_dir: str = None, test_count: int = 5, val_ratio: float = 0.2, seed: int = 42):
    """
    data_dir: path to preprocessed (normalized) epochs (contains subject_x folders)
    out_dir: where CSVs will be written (default = data_dir)
    test_count: number of subjects to hold-out as unseen test clients
    val_ratio: fraction of per-subject data used for validation (0.2 => 80/20)
    seed: random seed for reproducibility
    """
    random.seed(seed)
    data_dir = Path(data_dir).resolve()
    if out_dir is None:
        out_dir = data_dir
    else:
        out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect subject directories
    subjects = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    n_subjects = len(subjects)
    if n_subjects == 0:
        raise RuntimeError(f"No subject directories found under {data_dir}")

    # choose test subjects
    if test_count >= n_subjects:
        raise ValueError("test_count must be less than total number of subjects")
    test_subjects = random.sample(subjects, test_count)
    train_subjects = [s for s in subjects if s not in test_subjects]

    print(f"Found {n_subjects} subject folders. Using {len(test_subjects)} as unseen test clients.")
    print("Test subjects:", [s.name for s in test_subjects])

    all_train_rows = []
    all_val_rows = []
    all_test_rows = []

    # handle test subjects: put all their epochs into a test_list
    for subj in test_subjects:
        recs = gather_epochs(subj)
        if not recs:
            print(f"  WARNING: no epochs for test subject {subj.name}")
            continue
        # write per-subject test_list.csv
        write_list_csv(recs, subj / "test_list.csv", data_dir)
        all_test_rows += recs

    # handle training subjects: stratified split per subject
    for subj in train_subjects:
        recs = gather_epochs(subj)
        if not recs:
            print(f"  WARNING: no epochs for train subject {subj.name}")
            continue

        paths = [r[0] for r in recs]
        labels = [r[1] for r in recs]

        # if only one class present, do simple split without stratify
        try:
            train_idx, val_idx = train_test_split(
                list(range(len(paths))),
                test_size=val_ratio,
                random_state=seed,
                shuffle=True,
                stratify=labels if len(set(labels)) > 1 else None
            )
        except Exception as e:
            # fallback: non-stratified
            train_idx, val_idx = train_test_split(
                list(range(len(paths))),
                test_size=val_ratio,
                random_state=seed,
                shuffle=True
            )

        train_rows = [(paths[i], labels[i]) for i in train_idx]
        val_rows = [(paths[i], labels[i]) for i in val_idx]

        # write per-subject files
        write_list_csv(train_rows, subj / "train_list.csv", data_dir)
        write_list_csv(val_rows, subj / "val_list.csv", data_dir)

        all_train_rows += train_rows
        all_val_rows += val_rows

        print(f"Subject {subj.name}: total={len(paths)}, train={len(train_rows)}, val={len(val_rows)}")

    # write global aggregated lists
    write_list_csv(all_train_rows, out_dir / "all_train.csv", data_dir)
    write_list_csv(all_val_rows, out_dir / "all_val.csv", data_dir)
    write_list_csv(all_test_rows, out_dir / "all_test.csv", data_dir)

    print("\nGlobal CSVs written:")
    print(" -", (out_dir / "all_train.csv"))
    print(" -", (out_dir / "all_val.csv"))
    print(" -", (out_dir / "all_test.csv"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split per-subject normalized epochs into train/val and select unseen test subjects.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed normalized epochs (contains subject_x folders).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write CSVs (defaults to data_dir).")
    parser.add_argument("--test_count", type=int, default=5, help="Number of unseen test subjects to hold out.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio per subject (0.2 => 80/20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    split_subjects(args.data_dir, out_dir=args.out_dir, test_count=args.test_count, val_ratio=args.val_ratio, seed=args.seed)
