"""fed-dqn: A Flower / PyTorch app."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context: Context):
    num_rounds = 10
    fraction_fit = 1.0

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_available_clients=2,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
