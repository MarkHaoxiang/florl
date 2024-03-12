import argparse
import pathlib
import pickle as pkl

import numpy as np
from tqdm import tqdm

from .config_pendulum import *
from ..experiment_utils import *

path = pathlib.Path(__file__).parent
def main(save_path: str = "baseline_results.pkl", fixed_reset: bool = False):
    if fixed_reset:
        client_factory = fixed_client_factory
    else:
        client_factory = iid_client_factory

    baseline_results = []
    rng = np.random.default_rng(seed=SEED)
    for _ in range(EXPERIMENT_REPEATS):
        seed = rng.integers(0,65535) 
        client = client_factory.create_client(seed, config["rl"])

        # Manually run through the training loop
        hist_fit = []
        evaluation_reward = []
        for simulated_rounds in tqdm(range(TOTAL_ROUNDS)):
            _, metrics = client.train(config["fl"]["train_config"])
            hist_fit.append(metrics)
            evaluation_reward.append(client._evaluator.evaluate(client.policy, repeats=config["fl"]["evaluate_config"]["evaluation_repeats"]))

        baseline_results.append((hist_fit, evaluation_reward))

    if fixed_reset:
        save_path = "fixed_" + save_path
    pkl.dump(baseline_results, open(os.path.join(path, save_path), "wb"))


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fixed', action="store_true")
args = parser.parse_args()
if __name__ == "__main__":
    main(fixed_reset=args.fixed)
    