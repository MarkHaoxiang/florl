from flwr.server import ServerApp, ServerAppComponents, ServerConfig as FlwrServerConfig
from flwr.common import Context, Metrics
from flwr.server.strategy import FedAvg
from florl.common import Config
from florl.common.logging import metrics_aggregation_fn


class ServerConfig(Config):
    num_rounds: int
    fraction_fit: float


def fit_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]):
    # TODO: ignored for centralised training
    return {}


@metrics_aggregation_fn
def evaluation_metrics_aggregation_fn(
    # TODO: we need to change this
    # for our centralised example, we don't really care about the evaluation metrics
    # but we need to change this up for it to works properly
    results: list[int],
):
    episode_reward = float([x for x in results].mean())  # type: ignore
    metrics = {"episode_reward": episode_reward}

    print(metrics)
    return metrics  # type: ignore


def server_fn(context: Context, num_rounds: int, fraction_fit: float):
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        min_available_clients=1,
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
