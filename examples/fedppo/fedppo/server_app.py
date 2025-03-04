"""fed-dqn: A Flower / PyTorch app."""

import numpy as np
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def fit_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]):
    return {}  # TODO
    res: Metrics = {}
    total_samples = sum([metric[0] for metric in metrics])

    loss_metrics = ["loss_objective", "loss_critic", "loss_entropy"]
    for k in loss_metrics:
        res[k] = sum([metric[1][k] * metric[0] / total_samples for metric in metrics])  # type: ignore

    return res


def evaluation_metrics_aggregation_fn(results: list[tuple[int, Metrics]]) -> Metrics:
    return {}  # TODO
    episode_reward = float(np.array([x[1]["episode_reward"] for x in results]).mean())
    metrics = {"episode_reward": episode_reward}
    return metrics  # type: ignore


def server_fn(context: Context):
    num_rounds = 20
    fraction_fit = 1.0

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        min_available_clients=2,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluation_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
