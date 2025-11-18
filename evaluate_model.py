# evaluate_model.py
"""
Evaluate a saved EEGNet model on your test CSV and print metrics + save predictions.

Usage:
    python evaluate_model.py \
        --model server_saved_models/final_finetuned.pth \
        --data_dir normalized_epochs \
        --test_csv all_test.csv \
        --out predictions.csv \
        --batch_size 8
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score

from model_eegnet import EEGNet
from dataset_eeg import get_dataloader

def load_state_dict_from_file(agg_path: Path, device: torch.device):
    """
    Load a saved torch state_dict .pth file. Returns dict suitable for model.load_state_dict.
    """
    state = torch.load(str(agg_path), map_location=device)
    # If you saved a dict directly (state_dict) it should be loadable; otherwise try common wrappers
    if isinstance(state, dict):
        # Heuristics:
        if "state_dict" in state:
            return state["state_dict"]
        # If each value is a tensor (likely a true state_dict)
        if all(isinstance(v, torch.Tensor) for v in state.values()):
            return state
        # Some saving formats store nested dicts; try to find sub-dict that looks like a state_dict
        for k, v in state.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                return v
    raise RuntimeError(f"Unable to interpret saved file {agg_path} as a PyTorch state_dict. Inspect file.")

def evaluate(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    epoch_paths = []

    with torch.no_grad():
        for batch in loader:
            # dataset returns (x, y) where x shape (B,1,C,T)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x, y = batch["x"], batch["y"]
            else:
                raise RuntimeError("Unexpected batch format from DataLoader. Expected (x, y) tuple or dict.")

            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

            # Attempt to get epoch_path information if dataset exposes it:
            # Many datasets don't return path; in that case we skip epoch_path column.
            # If your DataLoader's dataset has a df with 'epoch_path' column, we can map by index.
            # Here we just append None.
            batch_size = preds.shape[0]
            epoch_paths.extend([None] * batch_size)

            y_true.extend(y_np.tolist())
            y_pred.extend(preds.tolist())
            y_probs.extend(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_probs), epoch_paths

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Build a test loader using get_dataloader.
    # get_dataloader returns (train_loader, val_loader) for a subject folder or (train_loader,val_loader) for parent dir.
    # We'll try to pass data_dir (parent) and the function will detect all_test.csv by default.
    # To make sure it uses the test CSV, we will try to call get_dataloader on the test CSV directly if provided.
    data_dir = Path(args.data_dir)

    # If user provided a test CSV filename under data_dir, build path and feed dataset directly.
    test_csv_path = data_dir / args.test_csv if args.test_csv else None

    if test_csv_path and test_csv_path.exists():
        # Create a DataLoader directly from the CSV - leverage EEGDataset via get_dataloader by passing the folder
        # Our get_dataloader chooses csv inside a folder; easiest approach: pass the folder that contains the CSV (data_dir)
        # but ensure it will pick the test CSV by looking for test_list.csv / all_test.csv - if not, user can pass explicit csv
        # So we will create dataset by constructing EEGDataset directly if necessary.
        # Easiest: call get_dataloader(data_dir) and then use val_loader if it points to test CSV.
        train_loader, val_loader = get_dataloader(str(data_dir), batch_size=args.batch_size, shuffle=False)
        # Prefer val_loader (which our get_dataloader sets to all_test/all_val searches)
        loader = val_loader if val_loader is not None else train_loader
    else:
        # just call get_dataloader on data_dir and pick val_loader if exists
        train_loader, val_loader = get_dataloader(str(data_dir), batch_size=args.batch_size, shuffle=False)
        loader = val_loader if val_loader is not None else train_loader

    # Infer channels & samples from loader first batch to build model skeleton
    it = iter(loader)
    first_batch = next(it)
    if isinstance(first_batch, (list, tuple)) and len(first_batch) >= 1:
        x_sample = first_batch[0]
    elif isinstance(first_batch, dict):
        x_sample = first_batch["x"]
    else:
        raise RuntimeError("Unexpected batch format when inferring dataset shape")

    # expected shape (B,1,C,T) or (B,C,T)
    if isinstance(x_sample, torch.Tensor):
        if x_sample.dim() == 4:
            _, d1, d2, d3 = x_sample.shape
            if d1 == 1:
                num_channels = int(d2)
                samples = int(d3)
            else:
                num_channels = int(d1)
                samples = int(d2) if d3 == 1 else int(d3)
        elif x_sample.dim() == 3:
            _, c, t = x_sample.shape
            num_channels = int(c)
            samples = int(t)
        else:
            raise RuntimeError(f"Unexpected sample tensor dim {x_sample.dim()} when inferring model shape")
    else:
        arr = np.asarray(x_sample)
        if arr.ndim == 4:
            _, d1, d2, d3 = arr.shape
            num_channels = int(d2) if d1 == 1 else int(d1)
            samples = int(d3)
        elif arr.ndim == 3:
            _, c, t = arr.shape
            num_channels = int(c)
            samples = int(t)
        else:
            raise RuntimeError("Unexpected numpy array shape for sample")

    print(f"Inferred num_channels={num_channels}, samples={samples}")

    # Build model and load state dict
    model = EEGNet(num_channels=num_channels, samples=samples, num_classes=args.num_classes)
    model.to(device)

    state_dict = load_state_dict_from_file(model_path, device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {model_path}")

    # Evaluate
    y_true, y_pred, y_probs, epoch_paths = evaluate(model, device, loader)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("=== Evaluation Summary ===")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    # Save predictions CSV
    out_rows = []
    for i in range(len(y_true)):
        out_rows.append({
            "epoch_path": epoch_paths[i] if epoch_paths[i] is not None else "",
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
            "prob_class0": float(y_probs[i][0]),
            "prob_class1": float(y_probs[i][1]) if y_probs.shape[1] > 1 else 0.0,
        })
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("server_saved_models\final_finetuned.pth", type=str, required=True, help="Path to model .pth (state_dict) to evaluate")
    parser.add_argument("normalized_epochs", type=str, default="normalized_epochs", help="Parent folder with subjects and CSVs")
    parser.add_argument("normalized_epochs/all_test.csv", type=str, default="all_test.csv", help="Test CSV filename inside data_dir (optional)")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV for predictions")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()
    main(args)
