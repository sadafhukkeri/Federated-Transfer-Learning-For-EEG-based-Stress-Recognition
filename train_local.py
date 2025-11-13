import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_eeg import EEGDataset
from model_eegnet import EEGNet

csv_path = "normalized_epochs/all_train.csv"
base_dir = "normalized_epochs"
dataset = EEGDataset(csv_path, base_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "eegnet_stress.pth")
print("Model saved")
