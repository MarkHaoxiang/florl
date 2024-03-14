Key Takeaways: Improves stability and prevents catastrophic loss. Potentially higher training speed.

NUM_CLIENTS = 5
TOTAL_ROUNDS = 100
FRAMES_PER_ROUND = 50
EXPERIMENT_REPEATS = 20
SEED = 0

Algorithm:
QT-OPT, RLFedAvg

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
            "capacity": max(128, TOTAL_ROUNDS * FRAMES_PER_ROUND)
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
            "evaluation_repeats": 1
        }
    }
})

Centralised Evaluation
