from typing import Any, List

import numpy as np
import flwr as fl


def get_federated_metrics(
    results: List[fl.server.History],
    experiment_repeats: int,
    num_clients: int,
    total_rounds: int,
    centralised_evaluation: bool = False,
):
    losses = np.array(
        [
            [x[1]["all"] for x in hist.metrics_distributed_fit["loss"]]
            for hist in results
        ]
    )
    losses = losses.transpose((0, 2, 1, 3)).reshape(
        (experiment_repeats * num_clients, total_rounds, 2)
    )[:, :, 1]

    if not centralised_evaluation:
        rewards = np.array(
            [
                [x[1]["all"] for x in hist.metrics_distributed["reward"]]
                for hist in results
            ]
        )
        rewards = rewards.transpose((0, 2, 1, 3)).reshape(
            (experiment_repeats * num_clients, total_rounds, 2)
        )[:, :, 1]
    else:
        rewards = np.array(
            [[x[1] for x in hist.metrics_centralized["reward"]][1:] for hist in results]
        )

    return losses, rewards


def plot_losses(
    ax, xs, losses: List[Any], label: str, color: str = "green", hatch="x", **kwargs
):
    losses_mean = losses.mean(axis=0)
    losses_std = losses.std(axis=0)
    ax.plot(xs, losses_mean, color=color, label=label, **kwargs)
    # for i in range(NUM_CLIENTS):
    #     ax.scatter(rounds, federated_losses[i], color="g", alpha=0.3, s=5)
    ax.fill_between(
        x=xs,
        y1=losses_mean - losses_std * 1.96,
        y2=losses_mean + losses_std * 1.96,
        alpha=0.2,
        color="white",
        facecolor=color,
        hatch=hatch,
    )


def plot_rewards(
    ax, xs, rewards: List[Any], label: str, color: str = "green", hatch="x", **kwargs
):
    rewards_mean = rewards.mean(axis=0)
    rewards_std = rewards.std(axis=0)

    ax.plot(xs, rewards_mean, color=color, label=label, **kwargs)
    # for i in range(NUM_CLIENTS):
    #     ax.scatter(rounds, federated_rewards[i], color="g", alpha=0.3, s=5)
    ax.fill_between(
        x=xs,
        y1=rewards_mean - rewards_std * 1.96,
        y2=rewards_mean + rewards_std * 1.96,
        alpha=0.2,
        color="white",
        hatch=hatch,
        facecolor=color,
        label=label
    )


def plot_fed_results(
    ax_loss, ax_reward, xs, results: List[Any], label: str, color: str = "green"
):
    losses, rewards = get_federated_metrics(results)
    plot_losses(
        ax_loss,
        xs,
    )
