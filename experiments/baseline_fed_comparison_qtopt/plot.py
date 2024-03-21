import os
import pathlib
import pickle as pkl

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(context="paper") 

from .config_pendulum import *
from ..experiment_utils import *
from ..visualisation import *

BASELINE_COLOR = "darkorange"
BASELINE_NAME = "QtOpt Centralised"
FEDERATED_COLOR = "blue"
FEDERATED_NAME = "QtOpt FedAvg"


path = pathlib.Path(__file__).parent
def main():
    # Collect Results
    baseline_results = pkl.load(open(os.path.join(path,"baseline_results.pkl"), "rb"))
    federated_results = pkl.load(open(os.path.join(path,"federated_results.pkl"), "rb"))
    fixed_baseline_results = pkl.load(open(os.path.join(path,"fixed_baseline_results.pkl"), "rb"))
    fixed_federated_results = pkl.load(open(os.path.join(path,"fixed_federated_results.pkl"), "rb"))

    baseline_losses = np.array([[s['loss'] for s in ex[0]] for ex in baseline_results])
    baseline_rewards = np.array([ex[1] for ex in baseline_results])
    fixed_baseline_losses = np.array([[s['loss'] for s in ex[0]] for ex in fixed_baseline_results])
    fixed_baseline_rewards = np.array([ex[1] for ex in fixed_baseline_results])
    federated_losses, federated_rewards = get_federated_metrics(
        federated_results,
        EXPERIMENT_REPEATS,
        NUM_CLIENTS,
        TOTAL_ROUNDS,
        centralised_evaluation=True
    )
    fixed_federated_losses, fixed_federated_rewards = get_federated_metrics(
        fixed_federated_results,
        EXPERIMENT_REPEATS,
        NUM_CLIENTS,
        TOTAL_ROUNDS,
        centralised_evaluation=True
    )

    # Figure Settings
    fig_l, axs_l = plt.subplots(1,2)
    fig_l.set_size_inches(11, 4)
    fig_l.suptitle("QTOpt/QTOptAvg (Pendulum)", weight="bold", fontsize=14)
    fig_r, axs_r = plt.subplots(1,2)
    fig_r.set_size_inches(11, 4)
    fig_r.suptitle("QTOpt/QTOptAvg (Pendulum)", weight="bold", fontsize=14)

    rounds = list(range(TOTAL_ROUNDS))

    # Losses
    titles = [
        "Training Loss (I.I.D Environment)",
        "Training Loss (Heterogenous Environment - Fixed Reset)"
    ]
    y_label = "Average Loss"
    x_label = "Round"
    baseline = [baseline_losses, fixed_baseline_losses]
    federated = [federated_losses, fixed_federated_losses]
    for i, ax in enumerate(axs_l):
        ax.set_title(titles[i])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        plot_losses(
            ax=ax,
            xs=rounds,
            losses=baseline[i],
            label=BASELINE_NAME,
            color=BASELINE_COLOR,
            linestyle="--",
            hatch="\\"
        )
        plot_losses(
            ax=ax,
            xs=rounds,
            losses=federated[i],
            label=FEDERATED_NAME,
            color=FEDERATED_COLOR,
            hatch="//"
        )

    handles, labels = axs_l[-1].get_legend_handles_labels()
    fig_l.legend(handles, labels, loc='lower right')

    # Rewards
    titles = [
        "Evaluation Reward (I.I.D Environment)",
        "Evaluation Reward (Heterogenous Environment - Fixed Reset)"
    ]
    y_label = "Average Episode Reward"
    x_label = "Round"
    baseline = [baseline_rewards, fixed_baseline_rewards]
    federated = [federated_rewards, fixed_federated_rewards]
    for i, ax in enumerate(axs_r):
        ax.set_title(titles[i])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        plot_rewards(
            ax=ax,
            xs=rounds,
            rewards=baseline[i],
            label=BASELINE_NAME,
            color=BASELINE_COLOR,
            linestyle="--",
            hatch="\\"
        )
        plot_rewards(
            ax=ax,
            xs=rounds,
            rewards=federated[i],
            label=FEDERATED_NAME,
            color=FEDERATED_COLOR,
            hatch="//"
        )
        for j in range(EXPERIMENT_REPEATS):
            ax.scatter(rounds, baseline[i][j, :], color=BASELINE_COLOR, alpha=0.1, s=3)
            ax.scatter(rounds, federated[i][j, :], color=FEDERATED_COLOR, alpha=0.1, s=3)

    handles, labels = axs_r[-1].get_legend_handles_labels()
    fig_r.legend(handles, labels, loc='lower right')

    # Save figures
    fig_l.savefig(os.path.join(path,"federating_qtopt_loss.png"), format="png", bbox_inches="tight")
    fig_l.savefig(os.path.join(path,"federating_qtopt_loss.pdf"), format="pdf", bbox_inches="tight")
    fig_r.savefig(os.path.join(path,"federating_qtopt_reward.png"), format="png", bbox_inches="tight")
    fig_r.savefig(os.path.join(path,"federating_qtopt_reward.pdf"), format="pdf", bbox_inches="tight")
#
if __name__ == "__main__":
    main()