"""fed-dqn: A Flower / PyTorch app."""

import numpy as np
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def evaluation_metrics_aggregation_fn(results: list[tuple[int, Metrics]]) -> Metrics:
    episode_reward = float(np.array([x[1]["episode_reward"] for x in results]).mean())
    metrics = {"episode_reward": episode_reward}
    return metrics  # type: ignore


def server_fn(context: Context):
    num_rounds = 10
    fraction_fit = 1.0

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=evaluation_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
