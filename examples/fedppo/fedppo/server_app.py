import numpy as np
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig as FlwrServerConfig
from flwr.server.strategy import FedAvg
from florl.common import Config
from florl.common.logging import JSONSerializable, metrics_aggregation_fn


class ServerConfig(Config):
    num_rounds: int
    fraction_fit: float


def fit_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]):
    return {}  # TODO
    res: Metrics = {}
    total_samples = sum([metric[0] for metric in metrics])

    loss_metrics = ["loss_objective", "loss_critic", "loss_entropy"]
    for k in loss_metrics:
        res[k] = sum([metric[1][k] * metric[0] / total_samples for metric in metrics])  # type: ignore

    return res


@metrics_aggregation_fn
def evaluation_metrics_aggregation_fn(
    results: list[tuple[int, JSONSerializable]],
):
    episode_reward = float(np.array([x[1]["episode_reward"] for x in results]).mean())
    metrics = {"episode_reward": episode_reward}

    print(metrics)
    return metrics  # type: ignore


def server_fn(context: Context, num_rounds: int, fraction_fit: float):
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        min_available_clients=2,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluation_metrics_aggregation_fn,
    )
    config = FlwrServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
def app(cfg: ServerConfig):
    def _server_fn(context: Context):
        return server_fn(context, cfg.num_rounds, cfg.fraction_fit)

    return ServerApp(server_fn=_server_fn)
