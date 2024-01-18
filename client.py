import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
# import tenseal as ts
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)


warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50
# NUM_CLIENTS = 5


# a config for mobilenetv2 that works for
# small input sizes (i.e. 32x32 as in CIFAR)
mb2_cfg = [
    (1, 16, 1, 1),
    (6, 24, 2, 1),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def prepare_dataset():

    """Get CIFAR-10 and return client partitions and global testset."""
    dataset = CIFAR10
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trf = Compose([ToTensor(), norm])
    trainset = dataset("./data", train=True, download=True, transform=trf)
    testset = dataset("./data", train=False, download=True, transform=trf)

    print("Partitioning dataset (IID)...")

    # Split trainset into `num_partitions` trainsets
    num_images = len(trainset) // NUM_CLIENTS
    partition_len = [num_images] * NUM_CLIENTS

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    val_ratio = 0.1

    # Create dataloaders with train+val support
    train_partitions = []
    val_partitions = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        train_partitions.append(for_train)
        val_partitions.append(for_val)

    return train_partitions, val_partitions, testset

# def encryption(params):
#     context = ts.Context()
#     context.generate_galois_keys()
    
#     # Encrypt each parameter using a TenSEAL CKKS encoder
#     encrypted_params = [ts.ckks_vector(context, p.tolist()) for p in params]
    
#     # Serialize encrypted parameters
#     encrypted_params_serialized = [ep.serialize() for ep in encrypted_params]
    
#     return encrypted_params_serialized

# def decryption(encrypted_params_serialized):
#     # Initialize TenSEAL context and keys
#     context = ts.Context()
#     context.generate_galois_keys()
    
#     # Deserialize encrypted parameters
#     encrypted_params = [ts.ckks_vector_from(context, ep) for ep in encrypted_params_serialized]
    
#     # Decrypt each parameter using a TenSEAL CKKS encoder
#     decrypted_params = [ep.decrypt() for ep in encrypted_params]
    
#     return decrypted_params


# Flower client
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10"""

    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model
        self.model = mobilenet_v3_small(num_classes=10)
        self.model.features[0][0].stride = (1, 1)
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        # Tenseal can decrypt here!
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)
        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    # Download CIFAR-10 dataset and partition it
    trainsets, valsets, _ = prepare_dataset()

    # Start Flower client setting its associated data partition
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid]
        ),
    )


if __name__ == "__main__":
    main()
