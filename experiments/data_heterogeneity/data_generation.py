import logging
import argparse
import shutil
import pickle as pkl
import pathlib

from tqdm import tqdm
import flwr as fl
from flwr.common.logger import logger
logger.setLevel(logging.WARNING)
from florl.common.util import aggregate_weighted_average, stateful_client
from florl.client.kitten.qt_opt import *

from .config import *
from ..experiment_utils import *

PATH = pathlib.Path(__file__).parent

class MemoryClientFactory(QTOptClientFactory):
    def __init__(self, config: Config, fixed_reset: bool = False, device: str = "cpu") -> None:
        super().__init__(config, device)
        if fixed_reset:
            self.env = FixedResetWrapper(self.env)

    def create_client(self, cid: str, config: Config, **kwargs) -> MemoryClient:
        client =  super().create_client(cid, config, **kwargs)
        return MemoryClient(client)

def main(fixed_reset: bool = False):
    client_factory = MemoryClientFactory(config, fixed_reset=fixed_reset)
    
    CONTEXT_WS = "florl_ws"

    evaluation_client = client_factory.create_client(0, config["rl"])._client
    strategy = RlFedAvg(
        knowledge=copy.deepcopy(client_factory.create_default_knowledge(config=config["rl"])),
        on_fit_config_fn = on_fit_config_fn,
        on_evaluate_config_fn= on_evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_weighted_average,
        evaluate_metrics_aggregation_fn=aggregate_weighted_average,
        evaluate_fn=get_evaluation_fn(evaluation_client),
        accept_failures=False,
        inplace=False
    )

    rng = np.random.default_rng(seed=SEED)
    for _ in tqdm(range(EXPERIMENT_REPEATS)):
        seed = rng.integers(0, 65535)
        if os.path.exists(CONTEXT_WS):
            shutil.rmtree(CONTEXT_WS)

        @stateful_client
        def build_client(cid: str) -> fl.client.Client:
            cid = int(cid) + seed
            return client_factory.create_client(
                cid=cid,
                config=config["rl"],
                enable_evaluation = False
            )

        hist = fl.simulation.start_simulation(
            client_fn=build_client,
            client_resources={'num_cpus': 1},
            config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
            num_clients = NUM_CLIENTS,
            strategy = strategy
        )

        if fixed_reset:
            path = os.path.join(PATH, "fixed", f"{str(seed)}.pkl")
        else:
            path = os.path.join(PATH, "iid", f"{str(seed)}.pkl")

        pkl.dump(hist, open(path, "wb"))

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fixed', action="store_true")
args = parser.parse_args()
if __name__ == "__main__":
    main(fixed_reset=args.fixed)