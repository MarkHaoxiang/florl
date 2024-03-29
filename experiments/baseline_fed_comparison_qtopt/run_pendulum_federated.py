import argparse
import logging
import pathlib
import pickle as pkl
import shutil
import copy
import pickle as pkl

import numpy as np
from tqdm import tqdm
import flwr as fl
from flwr.common.logger import logger
logger.setLevel(logging.WARNING)
from florl.common.util import aggregate_weighted_average, stateful_client

from .config_pendulum import *
from ..experiment_utils import *
from ..strategy import RlFedAvg

CONTEXT_WS = "florl_ws"

path = pathlib.Path(__file__).parent
def main(save_path: str = "federated_results.pkl", fixed_reset: bool = False):
    if fixed_reset:
        client_factory = fixed_client_factory
    else:
        client_factory = iid_client_factory

    strategy = RlFedAvg(
        knowledge=copy.deepcopy(client_factory.create_default_knowledge(config=config["rl"])),
        on_fit_config_fn = on_fit_config_fn,
        on_evaluate_config_fn= on_evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_weighted_average,
        evaluate_metrics_aggregation_fn=aggregate_weighted_average,
        evaluate_fn=get_evaluation_fn(client_factory.create_client(0, config["rl"])),
        accept_failures=False,
        inplace=False
    )

    federated_results = []
    rng = np.random.default_rng(seed=SEED)

    for _ in tqdm(range(EXPERIMENT_REPEATS)):
        seed = rng.integers(0, 65535)
        if os.path.exists(CONTEXT_WS):
            shutil.rmtree(CONTEXT_WS)

        initialized_clients = {}

        @stateful_client
        def build_client(cid: str) -> fl.client.Client:
            cid = int(cid) + seed
            if cid not in initialized_clients.keys():
                initialized_clients[cid] = client_factory.create_client(
                    cid=cid,
                    config=config["rl"],
                    enable_evaluation = False
                )
                return initialized_clients[cid]
            else:
                return initialized_clients[cid]

        hist = fl.simulation.start_simulation(
            client_fn=build_client,
            client_resources={'num_cpus': 1},
            config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
            num_clients = NUM_CLIENTS,
            strategy = strategy
        )
        
        federated_results.append(hist)

    if fixed_reset:
        save_path = "fixed_" + save_path
    pkl.dump(federated_results, open(os.path.join(path, save_path), "wb"))

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fixed', action="store_true")
args = parser.parse_args()
if __name__ == "__main__":
    main(fixed_reset=args.fixed)
    
    