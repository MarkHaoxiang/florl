from omegaconf import OmegaConf, DictConfig
from florl.client.kitten.qt_opt import QTOptClientFactory

from ..experiment_utils import *

NUM_CLIENTS = 5
TOTAL_ROUNDS = 200 
FRAMES_PER_ROUND = 50
EXPERIMENT_REPEATS = 20
SEED = 0

config = DictConfig({
    "rl": {
        "env": {
            "name": "Pendulum-v1"
        },
        "algorithm": {
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.001,
            "update_frequency": 1,
            "clip_grad_norm": 1,
            "critic": {
                "features": 64
            }
        },
        "memory": {
            "type": "experience_replay",
            "capacity": TOTAL_ROUNDS * FRAMES_PER_ROUND
        },
        "train": {
            "initial_collection_size": 1024,
            "minibatch_size": 64
        }
    },
    "fl": {
        "train_config": {
            "frames": FRAMES_PER_ROUND,
        },
        "evaluate_config": {
            "evaluation_repeats": 5
        }
    }
})

train_config = OmegaConf.to_container(config["fl"]["train_config"])
evaluate_config = OmegaConf.to_container(config["fl"]["evaluate_config"])

def on_fit_config_fn(server_round: int):
        return train_config | {"server_round": server_round}
def on_evaluate_config_fn(server_round: int):
    return evaluate_config | {"server_round": server_round}

class ClientFactory(QTOptClientFactory):
    def __init__(self, config, fixed_reset: bool = False, device: str = "cpu") -> None:
        super().__init__(config, device)
        if fixed_reset:
            self.env = FixedResetWrapper(self.env)

torch.manual_seed(SEED)
np.random.seed(SEED)
fixed_client_factory = ClientFactory(config, fixed_reset=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
iid_client_factory = ClientFactory(config)