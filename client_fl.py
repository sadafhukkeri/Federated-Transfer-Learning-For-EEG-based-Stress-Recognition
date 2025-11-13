import flwr as fl
import torch
from model_eegnet import EEGNet
from dataset_eeg import get_dataloader  

# Convert model parameters (weights) to a list of numpy arrays
def get_model_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Load parameters from the server into the local model
def set_model_params(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGClient(fl.client.NumPyClient):
    def __init__(self, subject_id):
        self.model = EEGNet(num_classes=2).to(DEVICE)
        self.trainloader = get_dataloader(f"normalized_epochs/subject_{subject_id:02d}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
    # Load the model parameters received from server
        set_model_params(self.model, parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1):  # 1 local epoch
            for data, target in self.train_loader:
                data = data.to(DEVICE)
                target = target.to(DEVICE)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return get_model_params(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.5, len(self.trainloader.dataset), {"accuracy": 0.9}  # dummy for now

if __name__ == "__main__":
    subject_id = 1  # change this for each client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=EEGClient(subject_id))

