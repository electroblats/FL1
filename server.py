import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

# To run:
# python3 server.py --rounds 3 --num_clients 2 --server_address=192.168.50.135:8080 --settings 0 --epochs 3 --batch 16

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
    "--min_num_fit",
    type=int,
    default=2,
    help="Minimum number of available clients required for fit (default: 2)",
)
parser.add_argument(
    "--min_num_evaluate",
    type=int,
    default=2,
    help="Minimum number of available clients required for evaluate (default: 2)",
)
# parser.add_argument(
#     "--min_available_clients",
#     type=int,
#     default=2,
#     help="Minimum number of available clients required to start FedAvg",
# )
parser.add_argument(
    "--num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required to start FL",
)
parser.add_argument(
    "--settings",
    type=int,
    default=0,
    help="advanced or simple sampling settings",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of local epochs done by clients",
)
parser.add_argument(
    "--batch",
    type=int,
    default=16,
    help="Batch size to use by clients during fit()",
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

# def fit_config(epochs=3,batch_size=16):
#     """Return a configuration with batch size and (local) epochs."""
#     config = {
#         "epochs": epochs,  # Number of local epochs done by clients
#         "batch_size": batch_size,  # Batch size to use by clients during fit()
#     }
#     return config


def main():
    args = parser.parse_args()

    print(args)

    if (args.settings==1):
        # Define strategy (advanced sampling options)
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_num_fit,
            min_evaluate_clients=args.min_num_evaluate,
            # min_available_clients=args.min_available_clients,
            min_available_clients=args.num_clients,
            on_fit_config_fn=fit_config,
            # on_fit_config_fn=fit_config(args.epochs,args.batch),
            evaluate_metrics_aggregation_fn=weighted_average,
            )
    else:
        # Define strategy (simple sampling options)
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            # min_available_clients=args.min_available_clients,
            min_available_clients=args.num_clients,
            on_fit_config_fn=fit_config,
            # on_fit_config_fn=fit_config(args.epochs,args.batch),
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    # # Waits for a specified number of clients, num_clients
    # client_manager = fl.server.client_manager.SimpleClientManager(
    #     wait_for = {
    #         "num_clients": args.num_clients,
    #         "timeout": 200,
    #     },
    # )

    # Start Flower server
    server = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        # client_manager=client_manager,
    )


if __name__ == "__main__":
    main()
