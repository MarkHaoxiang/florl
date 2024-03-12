import os
import pathlib
import pickle as pkl
import argparse

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(context="paper") 

from .config_pendulum import *
from ..experiment_utils import *
from ..visualisation import *



path = pathlib.Path(__file__).parent
def main(fixed_reset: bool = False):

    if not fixed_reset:
        baseline_results = pkl.load(open(os.path.join(path,"baseline_results.pkl"), "rb"))
        federated_results = pkl.load(open(os.path.join(path,"federated_results.pkl"), "rb"))
    else:
        baseline_results = pkl.load(open(os.path.join(path,"fixed_baseline_results.pkl"), "rb"))
        federated_results = pkl.load(open(os.path.join(path,"fixed_federated_results.pkl"), "rb"))

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(11, 4)
    fig.suptitle("QTOpt/QTOptAvg on Pendulum")

    plt.rcParams['svg.fonttype'] = 'none'
    FEDERATED_COLOR = "blue"
    BASELINE_COLOR = "darkorange"

    rounds = list(range(TOTAL_ROUNDS))

    # Loss & Rewards
    baseline_losses = np.array([[s['loss'] for s in ex[0]] for ex in baseline_results])
    baseline_rewards = np.array([ex[1] for ex in baseline_results])
    federated_losses, federated_rewards = get_federated_metrics(
        federated_results,
        EXPERIMENT_REPEATS,
        NUM_CLIENTS,
        TOTAL_ROUNDS,
        centralised_evaluation=True
    )


    ax_losses = axs[0]
    ax_losses.set_title("Training Loss (Distributed)")
    ax_losses.set_ylabel("Average Loss")
    ax_losses.set_xlabel("Round")
    # ax_losses.set_prop_cycle(color=['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet'])
    

    plot_losses(ax=ax_losses,
                xs=rounds,
                losses=baseline_losses,
                label="QtOpt Centralised",
                color=BASELINE_COLOR,
                linestyle="--",
                hatch="\\")

    plot_losses(ax=ax_losses,
                xs=rounds,
                losses=federated_losses,
                label="QtOpt FedAvg",
                color=FEDERATED_COLOR,
                hatch="//") 
    ax_losses.set_yscale("log")

    # Evaluation Reward
    ax_rewards = axs[1]
    ax_rewards.set_title("Evaluation Reward (Centralised)")
    ax_rewards.set_ylabel("Average Episode Reward")
    ax_rewards.set_xlabel("Round")

    for i in range(NUM_CLIENTS):
        ax_rewards.scatter(rounds, baseline_rewards[i, :], color=BASELINE_COLOR, alpha=0.3, s=5)
        ax_rewards.scatter(rounds, federated_rewards[i, :], color=FEDERATED_COLOR, alpha=0.3, s=5)

    plot_rewards(ax=ax_rewards,
                xs=rounds,
                rewards=baseline_rewards,
                label="QtOpt Centralised",
                color=BASELINE_COLOR,
                linestyle="--",
                hatch="\\")

    plot_rewards(ax=ax_rewards,
                xs=rounds,
                rewards=federated_rewards,
                label="QtOpt FedAvg",
                color=FEDERATED_COLOR,
                hatch="//")

    handles, labels = ax_rewards.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    #pickle.dump(fig, open('plot.pkl', 'wb'))
    #fig.savefig(os.path.join(path,"plot.svg"), format="svg", bbox_inches="tight")
    fig.savefig(os.path.join(path,"fplot.png"), format="png", bbox_inches="tight")
    fig.savefig(os.path.join(path,"fplot.pdf"), format="pdf", bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fixed', action="store_true")
args = parser.parse_args()
if __name__ == "__main__":
    main(fixed_reset=args.fixed)