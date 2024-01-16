import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--mnist",
    action="store_true",
    help="If you use Raspberry Pi Zero clients (which just have 512MB or RAM) use MNIST",
)
parser.add_argument(
    "--num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required to start FL",
)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Thist function averages teh `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,  # Number of local epochs done by clients
        "batch_size": 16,  # Batch size to use by clients during fit()
    }
    return config


def main():
    args = parser.parse_args()

    print(args)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Waits for a specified number of clients, num_clients
    # client_manager = fl.server.client_manager.SimpleClientManager.wait_for(num_clients=2, timeout=10)
    # client_manager = fl.server.SimpleClientManager.wait_for(num_clients=2, timeout=10)
    # ClientManager = fl.server.ClientManager.wait_for(num_clients=2, timeout=10)
    
    ClientManager = fl.server.client_manager.SimpleClientManager()
    client_manager = fl.server.client_manager.ClientManager.wait_for(ClientManager,num_clients=args.num_clients,timeout=10)

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_manager=client_manager,
        # client_manager=fl.server.client_manager.SimpleClientManager.wait_for(num_clients=args.num_clients),
    )
    # fl.server.ClientManager.wait_for(num_clients=args.num_clients,timeout=10)


if __name__ == "__main__":
    main()
