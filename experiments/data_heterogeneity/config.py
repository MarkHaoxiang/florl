from omegaconf import OmegaConf, DictConfig

NUM_CLIENTS = 4
TOTAL_ROUNDS = 50
FRAMES_PER_ROUND = 100 
EXPERIMENT_REPEATS = 20
SEED = 0

episode_length = FRAMES_PER_ROUND

config = DictConfig({
    "rl": {
        "env": {
            "name": "Pendulum-v1",
            "max_episode_steps": episode_length
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
            "initial_collection_size": episode_length,
            "minibatch_size": 64
        }
    },
    "fl": {
        "train_config": {
            "frames": FRAMES_PER_ROUND,
        },
        "evaluate_config": {
            "evaluation_repeats": 1
        }
    }
})

train_config = OmegaConf.to_container(config["fl"]["train_config"])
evaluate_config = OmegaConf.to_container(config["fl"]["evaluate_config"])

def on_fit_config_fn(server_round: int):
        return train_config | {"server_round": server_round}
def on_evaluate_config_fn(server_round: int):
    return evaluate_config | {"server_round": server_round}