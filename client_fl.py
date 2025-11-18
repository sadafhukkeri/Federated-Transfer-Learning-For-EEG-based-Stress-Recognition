# client_fl.py
# Fixed client: Flower compatibility (get_parameters/set_parameters accept optional config),
# delay model creation until after dataloader is known,
# auto-infer num_channels/samples if not provided, robust dataloader calling,
# optional Flower client startup via CLI.

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Flower import — allow running without Flower installed for local debugging
try:
    import flwr as fl
except Exception:
    fl = None

# Project imports (adjust if your project layout differs)
from dataset_eeg import get_dataloader  # flexible call handled below
from model_eegnet import EEGNet


# -------------------------
# Helpers to convert params
# -------------------------
def get_model_params(model: torch.nn.Module) -> List[np.ndarray]:
    """Return model parameters as a list of NumPy arrays (compatible with Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays (compatible with Flower)."""
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(
            f"Length mismatch: got {len(params)} param arrays but model expects {len(state_dict)}"
        )
    new_state = {}
    for (k, v), p in zip(state_dict.items(), params):
        tensor = torch.tensor(p, dtype=v.dtype)
        new_state[k] = tensor
    model.load_state_dict(new_state)


# -------------------------
# EEG Federated Client
# -------------------------
class EEGClient(fl.client.NumPyClient if fl is not None else object):
    """
    Flower NumPyClient that trains locally on the client's EEG data.
    Automatically resolves subject folder and calls get_dataloader robustly.
    Model creation is deferred until the dataloader is available so we can infer
    input channel/time dimensions and avoid mismatch errors.
    """

    def __init__(
        self,
        subject_id: int,
        data_dir: str,
        model_kwargs: Optional[Dict] = None,
        lr: float = 1e-3,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        subject_id: integer id of the subject (1..40)
        data_dir: either
            - path to the parent folder containing subject_xx folders, e.g. "normalized_epochs"
            - OR path to a specific subject folder, e.g. "normalized_epochs/subject_01"
        model_kwargs: dict forwarded to EEGNet (may omit num_channels/samples to infer automatically)
        lr: local learning rate
        device: "cpu" or "cuda" or None (auto-detect)
        batch_size: dataloader batch size to use when querying dataset shapes
        """
        self.subject_id = int(subject_id)
        self.data_dir = str(data_dir)
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = dict(model_kwargs) if model_kwargs else {}
        self.batch_size = int(batch_size)

        # Placeholders — actual model created after dataloader is built
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[torch.nn.Module] = None

        # --- Resolve subject folder path ---
        data_path = Path(self.data_dir)
        # If user passed a parent folder like "normalized_epochs", build subject folder
        if "subject" in data_path.name.lower():
            # They passed subject folder already
            subject_path = data_path
            parent_dir = data_path.parent
        else:
            # Build "subject_XX" folder inside provided parent
            subject_folder_name = f"subject_{self.subject_id:02d}"
            subject_path = data_path / subject_folder_name
            parent_dir = data_path

        # If subject_path doesn't exist, try alternative: maybe they passed "normalized_epochs/subject_1" (no zero-pad)
        if not subject_path.exists():
            alt_folder = data_path / f"subject_{self.subject_id}"
            if alt_folder.exists():
                subject_path = alt_folder

        # If still not exists, raise helpful message
        if not subject_path.exists():
            raise FileNotFoundError(
                f"Subject folder not found:\n  tried: {subject_path}\n\n"
                f"Make sure you pass data_dir='normalized_epochs' (parent) or data_dir='normalized_epochs/subject_01' (subject folder),\n"
                f"and that the folder for subject {self.subject_id} exists."
            )

        # --- Try calling get_dataloader in several ways (different possible signatures) ---
        train_loader = None
        val_loader = None
        last_exception = None

        # Candidate call forms to try, in order
        call_attempts = []

        # 1) keyword arg 'subject_path' (some implementations)
        call_attempts.append(("subject_path_kw", {"subject_path": subject_path, "batch_size": self.batch_size, "shuffle": True}))
        # 2) positional Path object
        call_attempts.append(("subject_path_pos", (subject_path,)))
        # 3) positional string path
        call_attempts.append(("subject_path_pos_str", (str(subject_path),)))
        # 4) call with parent data_dir (some get_dataloader implementations expect parent)
        call_attempts.append(("data_dir_pos", (str(data_path),)))
        call_attempts.append(("data_dir_pos_path", (data_path,)))

        for name, args in call_attempts:
            try:
                if isinstance(args, dict):
                    res = get_dataloader(**args)
                else:
                    # If args is a tuple, expand and include batch_size/shuffle where relevant
                    if isinstance(args, tuple) and len(args) == 1:
                        res = get_dataloader(args[0], batch_size=self.batch_size, shuffle=True)
                    else:
                        res = get_dataloader(*args)
                # Expecting (train_loader, val_loader) or single loader
                if isinstance(res, tuple) and len(res) >= 1:
                    train_loader, val_loader = res[0], (res[1] if len(res) > 1 else None)
                else:
                    train_loader = res
                    val_loader = None
                break
            except TypeError as te:
                last_exception = te
                # Try next signature
                continue
            except FileNotFoundError as fe:
                # Propagate because it indicates a missing CSV/epochs under candidate path
                raise FileNotFoundError(
                    f"get_dataloader raised FileNotFoundError when called with {name} -> {args!r}.\n"
                    f"Underlying message: {fe}\n\n"
                    f"Check that the expected CSVs (e.g., train_list.csv or all_train.csv) and .npy epoch files exist under: {subject_path}"
                ) from fe
            except Exception as e:
                last_exception = e
                continue

        if train_loader is None:
            # None of the call attempts worked: raise helpful error including last exception
            raise RuntimeError(
                "Failed to create dataloaders for subject.\n"
                f"Subject folder: {subject_path}\n"
                f"Attempted calling get_dataloader with several signatures; last exception:\n{last_exception}"
            )

        # store loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # --- Now that we have a loader, infer shape and create model if not provided ---
        # If model_kwargs does not define num_channels or samples, infer from one batch
        inferred_num_channels = None
        inferred_samples = None

        # try to peek a single batch safely
        try:
            it = iter(self.train_loader)
            batch = next(it)
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x_sample = batch[0]
            elif isinstance(batch, dict):
                x_sample = batch["x"]
            else:
                raise RuntimeError("Unexpected batch format when inferring dataset shape")

            # x_sample shape expected: (batch, 1, channels, samples) after dataset fix
            if isinstance(x_sample, torch.Tensor):
                if x_sample.dim() == 4:
                    # (B, 1, C, T) or (B, C, T, 1) — we expect (B,1,C,T)
                    # Defensive: prefer (B,1,C,T)
                    b, d1, d2, d3 = x_sample.shape
                    if d1 == 1:
                        inferred_num_channels = int(d2)
                        inferred_samples = int(d3)
                    else:
                        # If leading channel not 1, try to infer by shape heuristics
                        inferred_num_channels = int(d1)
                        inferred_samples = int(d2) if d3 == 1 else int(d3)
                elif x_sample.dim() == 3:
                    # (B, C, T) -> treat as (B,1,C,T) with implicit channel
                    b, c, t = x_sample.shape
                    inferred_num_channels = int(c)
                    inferred_samples = int(t)
                else:
                    raise RuntimeError(f"Unexpected sample tensor dim {x_sample.dim()} when inferring model shape")
            else:
                # if not tensor, try numpy
                xarr = np.asarray(x_sample)
                if xarr.ndim == 4:
                    _, d1, d2, d3 = xarr.shape
                    inferred_num_channels = int(d2) if d1 == 1 else int(d1)
                    inferred_samples = int(d3)
                elif xarr.ndim == 3:
                    _, c, t = xarr.shape
                    inferred_num_channels = int(c)
                    inferred_samples = int(t)
        except StopIteration:
            raise RuntimeError("Train loader is empty — cannot infer dataset shape")
        except Exception as e:
            raise RuntimeError(f"Failed to infer dataset shape from train loader: {e}")

        # Apply inferred values if not provided in model_kwargs
        if "num_channels" not in self.model_kwargs or self.model_kwargs.get("num_channels") is None:
            self.model_kwargs["num_channels"] = inferred_num_channels
        if "samples" not in self.model_kwargs or self.model_kwargs.get("samples") is None:
            self.model_kwargs["samples"] = inferred_samples
        if "num_classes" not in self.model_kwargs or self.model_kwargs.get("num_classes") is None:
            # default to 2 classes if unknown (you can override)
            self.model_kwargs["num_classes"] = self.model_kwargs.get("num_classes", 2)

        # Create model now that dimensions are known
        self.model = EEGNet(**self.model_kwargs).to(self.device)

        # Optimizer & loss (now that model exists)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # Flower-compatible: accept optional config argument (some Flower wrappers pass config)
    def get_parameters(self, config: Optional[Dict] = None) -> List[np.ndarray]:
        """Return the current model parameters as a list of NumPy arrays (Flower's NumPyClient API)."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        return get_model_params(self.model)

    # Flower-compatible: accept optional config argument
    def set_parameters(self, parameters: List[np.ndarray], config: Optional[Dict] = None) -> None:
        """Set model parameters from a list of NumPy arrays (Flower's NumPyClient API)."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        set_model_params(self.model, parameters)

    # Flower: local training
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Perform local training and return:
            - updated parameters (list of numpy arrays)
            - number of training examples used
            - optional metrics dict
        """
        # Set incoming global parameters
        self.set_parameters(parameters)

        # local epochs: may be provided by server via config, else default 1
        local_epochs = int(config.get("local_epochs", 1)) if config is not None else 1
        self.model.train()

        train_examples = 0
        total_loss = 0.0

        for epoch in range(local_epochs):
            for batch in self.train_loader:
                # Support dataset that returns either (x, y) or dict-like
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                elif isinstance(batch, dict):
                    x, y = batch["x"], batch["y"]
                else:
                    raise RuntimeError("Unexpected batch format from DataLoader. Expected (x, y) tuple or dict.")

                # Move to device
                x = x.to(self.device)
                y = y.to(self.device).long()

                # forward / backward
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                batch_size = x.shape[0]
                train_examples += batch_size
                total_loss += loss.item() * batch_size

        avg_loss = total_loss / max(1, train_examples)

        # Return updated parameters, number of examples, and optional metrics
        return get_model_params(self.model), train_examples, {"loss": float(avg_loss)}

    # Flower: evaluation on local validation set
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on client validation data.
        Returns:
            - loss (float)
            - number of evaluation examples (int)
            - metrics dict (e.g., {"accuracy": value})
        """
        # Set incoming parameters
        self.set_parameters(parameters)
        self.model.eval()

        if self.val_loader is None:
            # If no validation loader provided, attempt to use training loader for a quick check
            loader = self.train_loader
        else:
            loader = self.val_loader

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                elif isinstance(batch, dict):
                    x, y = batch["x"], batch["y"]
                else:
                    raise RuntimeError("Unexpected batch format from DataLoader. Expected (x, y) tuple or dict.")

                x = x.to(self.device)
                y = y.to(self.device).long()

                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * x.shape[0]

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.shape[0]

        eval_examples = total
        avg_loss = total_loss / max(1, eval_examples)
        accuracy = correct / max(1, eval_examples) if eval_examples > 0 else 0.0

        return float(avg_loss), eval_examples, {"accuracy": float(accuracy)}


# Optional helper to start a Flower client process from this file (example)
def start_flower_client(subject_id: int, data_dir: str, model_kwargs: dict, server_address: str = "localhost:8080", batch_size: int = 32):
    """
    Convenience function to start a Flower client when using Flower.
    Example:
        start_flower_client(subject_id=1, data_dir="normalized_epochs", model_kwargs={"num_channels":32,"samples":3200,"num_classes":2})
    """
    if fl is None:
        raise RuntimeError("Flower (flwr) is not installed. Install it or run client code manually.")

    client = EEGClient(subject_id=subject_id, data_dir=data_dir, model_kwargs=model_kwargs, batch_size=batch_size)
    fl.client.start_numpy_client(server_address=server_address, client=client)


# Example usage for local debugging or connecting to server:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1, help="Subject id (1..40)")
    parser.add_argument("--data_dir", type=str, default="normalized_epochs", help="Parent or subject folder")
    parser.add_argument("--server", type=str, default=None, help="If provided, connect to Flower server at address (e.g. localhost:8080). If omitted, run a local train test.")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local epochs for quick test (only used for local test).")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    SUBJECT_ID = args.subject
    DATA_DIR = args.data_dir
    # allow the client to infer num_channels/samples if you leave model_kwargs empty
    MODEL_KW = {}  # e.g., {"num_channels":33,"samples":3200,"num_classes":2}  (optional)

    if args.server:
        # Start a Flower client which connects to server and participates in federated rounds
        print(f"Starting Flower client for subject {SUBJECT_ID} connecting to {args.server}")
        client = EEGClient(subject_id=SUBJECT_ID, data_dir=DATA_DIR, model_kwargs=MODEL_KW, batch_size=args.batch_size)
        import flwr as fl  # ensure flwr is installed
        fl.client.start_numpy_client(server_address=args.server, client=client)
    else:
        # Run a quick local training test (same as before) and exit
        client = EEGClient(subject_id=SUBJECT_ID, data_dir=DATA_DIR, model_kwargs=MODEL_KW, batch_size=args.batch_size)
        params = client.get_parameters()
        updated_params, n_examples, metrics = client.fit(params, {"local_epochs": args.local_epochs})
        print("Trained on examples:", n_examples, "metrics:", metrics)
        loss, n_eval, eval_metrics = client.evaluate(updated_params, {})
        print("Eval loss:", loss, "examples:", n_eval, "metrics:", eval_metrics)
