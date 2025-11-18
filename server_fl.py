# server_fl.py
"""
Robust Flower server script for EEG federated training.
- Handles cases where FedAvg.aggregate_fit returns (Parameters, metrics) or Parameters directly.
- Safely extracts Parameters, converts to numpy arrays, maps them into a torch state_dict
  using EEGNet state keys, and saves aggregated_round_{rnd}.pth.
- After federated rounds, runs server-side transfer learning on the final aggregated .pth.
"""

import os
from pathlib import Path
from typing import Optional, Any

import flwr as fl
import numpy as np
import torch
from flwr.server.strategy import FedAvg
from flwr.common import Parameters

from model_eegnet import EEGNet
import transfer_learning

# -----------------------
# CONFIG - update if needed
# -----------------------
NUM_CHANNELS = 33     # e.g., 33
SAMPLES = 3200        # e.g., 3200
NUM_CLASSES = 2
NUM_ROUNDS = 3
OUTPUT_DIR = Path("server_saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------
# Utility: extract a Parameters object from possibly wrapped return values
# -----------------------
def extract_parameters(obj: Any) -> Optional[Parameters]:
    """
    Given obj which may be:
      - a flwr.common.Parameters instance
      - a tuple/list like (Parameters, metrics) or (None, ...)
      - a dict containing 'parameters' key
    Return the Parameters instance or None if none found.
    """
    if obj is None:
        return None

    # If it's already a Parameters instance
    if isinstance(obj, Parameters):
        return obj

    # If it's a tuple/list, try to find a Parameters element
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if isinstance(item, Parameters):
                return item
        # Some Flower versions return (Parameters, fit_res) where Parameters is wrapped inside a dict-like object.
        # Try nested search:
        for item in obj:
            if isinstance(item, (tuple, list, dict)):
                found = extract_parameters(item)
                if found is not None:
                    return found

    # If it's a dict, maybe has 'parameters' or 'weights' key
    if isinstance(obj, dict):
        if "parameters" in obj and isinstance(obj["parameters"], Parameters):
            return obj["parameters"]
        # maybe raw ndarrays stored directly
        if "parameters" in obj and not isinstance(obj["parameters"], Parameters):
            # can't convert, return None
            return None

    # Nothing found
    return None


# -----------------------
# Helper: save aggregated parameters -> torch .pth using model state_dict keys
# -----------------------
def save_parameters_to_torch_state(parameters_obj: Parameters, model: torch.nn.Module, filename: str) -> None:
    """
    Convert Flower Parameters -> ndarray list -> map to model.state_dict keys -> save as torch .pth.
    """
    if parameters_obj is None:
        raise ValueError("parameters_obj is None")

    # Convert to list of numpy arrays
    ndarrays = fl.common.parameters_to_ndarrays(parameters_obj)

    # Model keys
    state_keys = list(model.state_dict().keys())

    if len(ndarrays) != len(state_keys):
        # Diagnostic info for debugging mismatches
        raise RuntimeError(
            f"Length mismatch: {len(ndarrays)} ndarrays vs {len(state_keys)} model state keys.\n"
            f"First few state keys: {state_keys[:10]}"
        )

    state_dict = {}
    for k, arr in zip(state_keys, ndarrays):
        state_dict[k] = torch.tensor(arr)

    torch.save(state_dict, filename)
    print(f"[SERVER] Saved aggregated torch state_dict to: {filename}")


# -----------------------
# Custom FedAvg that saves aggregated model each round
# -----------------------
class SaveAndTransferFedAvg(FedAvg):
    def aggregate_fit(self, rnd: int, results, failures) -> Optional[Parameters]:
        """
        Call parent aggregation, then extract the actual Parameters (if wrapped) and save as .pth.
        """
        aggregated = super().aggregate_fit(rnd, results, failures)

        # aggregated may be Parameters or (Parameters, metrics) etc.
        params_obj = extract_parameters(aggregated)
        if params_obj is None:
            print(f"[SERVER] Round {rnd}: No Parameters object could be extracted from aggregate result; skipping save.")
            return aggregated

        # Build model skeleton matching clients
        model_skel = EEGNet(num_channels=NUM_CHANNELS, samples=SAMPLES, num_classes=NUM_CLASSES)

        out_path = OUTPUT_DIR / f"aggregated_round_{rnd}.pth"
        try:
            save_parameters_to_torch_state(params_obj, model_skel, str(out_path))
        except Exception as e:
            print(f"[SERVER] Warning: failed to save aggregated parameters as torch state: {e}")
            # Fallback: save raw ndarrays to .npy
            try:
                raw = fl.common.parameters_to_ndarrays(params_obj)
                np.save(str(OUTPUT_DIR / f"aggregated_round_{rnd}.npy"), raw)
                print(f"[SERVER] Saved fallback numpy arrays to {OUTPUT_DIR / f'aggregated_round_{rnd}.npy'}")
            except Exception as e2:
                print(f"[SERVER] Failed fallback save as well: {e2}")

        return aggregated


# -----------------------
# Main: start server and run transfer learning after finished
# -----------------------
def main():
    strategy = SaveAndTransferFedAvg()
    print("[SERVER] Starting Flower server...")
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("[SERVER] Federated training finished. Server history:", history)

    # Try to load last aggregated round's .pth and run transfer learning
    last_file = OUTPUT_DIR / f"aggregated_round_{NUM_ROUNDS}.pth"
    if last_file.exists():
        print(f"[SERVER] Found aggregated model {last_file}; starting server-side transfer learning.")
        try:
            transfer_learning.fine_tune_aggregated_model(
                agg_model_path=str(last_file),
                val_csv="normalized_epochs/all_val.csv",
                base_dir="normalized_epochs",
                num_epochs=3,
                lr=1e-4,
                device=None,
                save_path=str(OUTPUT_DIR / "final_finetuned.pth"),
            )
            print("[SERVER] Transfer learning finished and saved final_finetuned.pth")
        except Exception as e:
            print(f"[SERVER] Transfer learning failed: {e}")
    else:
        # Fallback: check for numpy fallback
        fallback = OUTPUT_DIR / f"aggregated_round_{NUM_ROUNDS}.npy"
        if fallback.exists():
            print(f"[SERVER] Found fallback numpy aggregated parameters: {fallback}.")
            print("[SERVER] To run transfer learning, convert the numpy list into a torch state_dict using the model skeleton,")
            print("         or re-run server with a matching model architecture so .pth is saved.")
        else:
            print(f"[SERVER] No aggregated model found for round {NUM_ROUNDS}. Skipping transfer learning.")


if __name__ == "__main__":
    main()
