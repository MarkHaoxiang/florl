import os
import pathlib
import pickle as pkl
import argparse

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(context="paper") 

from .config import *
from ..experiment_utils import *
from ..visualisation import *



path = pathlib.Path(__file__).parent
def main():

    baseline_results = pkl.load(open(os.path.join(path,"baseline_results.pkl"), "rb"))
    federated_results = pkl.load(open(os.path.join(path,"federated_results.pkl"), "rb"))
    fixed_baseline_results = pkl.load(open(os.path.join(path,"fixed_baseline_results.pkl"), "rb"))
    fixed_federated_results = pkl.load(open(os.path.join(path,"fixed_federated_results.pkl"), "rb"))

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(11, 4)
    fig.suptitle("DQN/DQNAvg", weight="bold", fontsize=14)

    plt.rcParams['svg.fonttype'] = 'none'
    FEDERATED_COLOR = "blue"
    BASELINE_COLOR = "darkorange"

    rounds = list(range(TOTAL_ROUNDS))

    # Loss & Rewards
    baseline_losses = np.array([[s['loss'] for s in ex[0]] for ex in baseline_results])
    baseline_rewards = np.array([ex[1] for ex in baseline_results])
    fixed_baseline_losses = np.array([[s['loss'] for s in ex[0]] for ex in fixed_baseline_results])
    fixed_baseline_rewards = np.array([ex[1] for ex in fixed_baseline_results])
    # federated_losses, federated_rewards = get_federated_metrics(
    #     federated_results,
    #     EXPERIMENT_REPEATS,
    #     NUM_CLIENTS,
    #     TOTAL_ROUNDS,
    #     centralised_evaluation=True
    # )
    fixed_federated_losses, fixed_federated_rewards = get_federated_metrics(
        fixed_federated_results,
        EXPERIMENT_REPEATS,
        NUM_CLIENTS,
        TOTAL_ROUNDS,
        centralised_evaluation=True
    )

    # ax_losses = axs[0]
    # ax_losses.set_title("Training Loss (I.I.D Environment)")
    # ax_losses.set_ylabel("Average Loss")
    # ax_losses.set_xlabel("Round")
    # ax_losses.set_prop_cycle(color=['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet'])

#    plot_losses(ax=ax_losses,
#                xs=rounds,
#                losses=baseline_losses,
#                label="DQN Centralised",
#                color=BASELINE_COLOR,
#                linestyle="--",
#                hatch="\\")
#    plot_losses(ax=ax_losses,
#                xs=rounds,
#                losses=federated_losses,
#                label="DQN FedAvg",
#                color=FEDERATED_COLOR,
#                hatch="//") 
#     # ax_losses.set_yscale("log")
    ax_losses = axs[0]
    ax_losses.set_title("Training Loss (Heterogenous Environment - Fixed Reset)")
    ax_losses.set_ylabel("Average Loss")
    ax_losses.set_xlabel("Round")
    # ax_losses.set_prop_cycle(color=['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet'])
    

    plot_losses(ax=ax_losses,
                xs=rounds,
                losses=fixed_baseline_losses,
                label="DQN Centralised",
                color=BASELINE_COLOR,
                linestyle="--",
                hatch="\\")
    plot_losses(ax=ax_losses,
                xs=rounds,
                losses=fixed_federated_losses,
                label="DQN FedAvg",
                color=FEDERATED_COLOR,
                hatch="//") 
#    ax_losses.set_yscale("log")


     # Evaluation Reward
    #ax_rewards = axs[1]
    #ax_rewards.set_title("Evaluation Reward (I.I.D Environment)")
    #ax_rewards.set_ylabel("Average Episode Reward")
    #ax_rewards.set_xlabel("Round")

    #for i in range(EXPERIMENT_REPEATS):
    #    ax_rewards.scatter(rounds, baseline_rewards[i, :], color=BASELINE_COLOR, alpha=0.1, s=3)
    #    ax_rewards.scatter(rounds, federated_rewards[i, :], color=FEDERATED_COLOR, alpha=0.1, s=3)
#
    #plot_rewards(ax=ax_rewards,
    #            xs=rounds,
    #            rewards=baseline_rewards,
    #            label="DQN Centralised",
    #            color=BASELINE_COLOR,
    #            linestyle="--",
    #            hatch="\\")
#
    #plot_rewards(ax=ax_rewards,
    #            xs=rounds,
    #            rewards=federated_rewards,
    #            label="DQN FedAvg",
    #            color=FEDERATED_COLOR,
    #            hatch="//")
# 
    # Evaluation Reward
    ax_rewards = axs[1]
    ax_rewards.set_title("Evaluation Reward (Heterogenous Environment - Fixed Reset)")
    ax_rewards.set_ylabel("Average Episode Reward")
    ax_rewards.set_xlabel("Round")

    for i in range(EXPERIMENT_REPEATS):
        ax_rewards.scatter(rounds, fixed_baseline_rewards[i, :], color=BASELINE_COLOR, alpha=0.1, s=3)
        ax_rewards.scatter(rounds, fixed_federated_rewards[i, :], color=FEDERATED_COLOR, alpha=0.1, s=3)

    plot_rewards(ax=ax_rewards,
                xs=rounds,
                rewards=fixed_baseline_rewards,
                label="DQN Centralised",
                color=BASELINE_COLOR,
                linestyle="--",
                hatch="\\")

    plot_rewards(ax=ax_rewards,
                xs=rounds,
                rewards=fixed_federated_rewards,
                label="DQN FedAvg",
                color=FEDERATED_COLOR,
                hatch="//")

    handles, labels = ax_rewards.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')

    # ##pickle.dump(fig, open('plot.pkl', 'wb'))
    # ##fig.savefig(os.path.join(path,"plot.svg"), format="svg", bbox_inches="tight")
    fig.savefig(os.path.join(path,"plot.png"), format="png", bbox_inches="tight")
    fig.savefig(os.path.join(path,"plot.pdf"), format="pdf", bbox_inches="tight")
#
if __name__ == "__main__":
    main()