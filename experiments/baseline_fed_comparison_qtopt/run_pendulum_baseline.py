import sys
import pickle as pkl

import numpy as np
import tqdm as tqdm

from .config_pendulum import *

def main(save_path: str = "baseline_results.pkl"):
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

    pkl.dump(baseline_results, open(save_path, "wb"))

if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) > 1:
        path = arguments[1]
    else:
        path = "baseline_results.pkl"
    main(save_path=path)
    