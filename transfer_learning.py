# transfer_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataset_eeg import get_dataloader, EEGDataset  # your dataset file
from model_eegnet import EEGNet
import os

def load_state_dict_from_numpy_params(params, model):
    """
    Params: list of numpy arrays returned by Flower or loaded via saved torch state_dict.
    If you saved as state_dict.pth (torch.save(model.state_dict())), simply use load_state_dict.
    This helper is for the case you saved a full torch state dict already.
    """
    model.load_state_dict(params)

def fine_tune_aggregated_model(agg_model_path: str,
                               val_csv: str = "normalized_epochs/all_val.csv",
                               base_dir: str = "normalized_epochs",
                               num_epochs: int = 5,
                               lr: float = 1e-4,
                               device: str = None,
                               save_path: str = "final_finetuned.pth"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    agg_model_path = Path(agg_model_path)
    if not agg_model_path.exists():
        raise FileNotFoundError(f"Aggregated model file not found: {agg_model_path}")

    # Load a model skeleton (must match the aggregated model's architecture)
    # If your aggregated model was EEGNet with num_channels and samples, supply them:
    # Option: try to infer channels/samples from a small val loader (robust).
    # Here we create a val_loader first to infer shapes.
    val_loader, _ = get_dataloader(base_dir, batch_size=8, shuffle=False)  # try parent; get_dataloader is robust
    # pick first batch to infer channels/samples
    import numpy as np
    it = iter(val_loader)
    x, y = next(it)
    b, ch_dim1, ch, samp = x.shape if x.ndim == 4 else (None, None, None, None)
    # assume ch is channels dimension when shape is (B,1,C,T)
    num_channels = ch
    samples = samp

    # Build model and load aggregated weights
    model = EEGNet(num_channels=num_channels, samples=samples, num_classes=2)
    # aggregated file could be a torch state_dict; load accordingly
    state = torch.load(str(agg_model_path), map_location=device)
    # support both state_dict and wrapped dict
    if isinstance(state, dict) and all(isinstance(v, (torch.Tensor,)) for v in state.values()):
        model.load_state_dict(state)
    else:
        # if you saved {"state_dict": state_dict}
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            # fallback: assume it's directly compatible
            try:
                model.load_state_dict(state)
            except Exception as e:
                raise RuntimeError(f"Unable to load aggregated model file: {e}")

    model.to(device)

    # Replace classifier to fine-tune only classifier (optional)
    # Example: freeze feature extractor layers and only train final classifier
    for name, param in model.named_parameters():
        if "classify" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # If you want to fine-tune entire model, uncomment:
    # for p in model.parameters(): p.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Use val_loader as "server-side fine-tune dataset" — if you have a dedicated finetune set, use it
    for epoch in range(num_epochs):
        model.train()
        total = 0
        running_loss = 0.0
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device).long()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += xb.size(0)
            running_loss += loss.item() * xb.size(0)
        print(f"[Transfer] Epoch {epoch+1}/{num_epochs} loss: {running_loss/total:.4f}")

    # Save final fine-tuned model
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")
    return save_path


if __name__ == "__main__":
    # Example usage if you run this file directly:
    # python transfer_learning.py
    agg = "aggregated_global.pth"
    fine_tune_aggregated_model(agg_model_path=agg, num_epochs=3, lr=1e-4, save_path="final_finetuned.pth")
