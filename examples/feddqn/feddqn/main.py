import os

from flwr.simulation import run_simulation
from florl.common import Config
from omegaconf import DictConfig
import hydra

from feddqn.task import TaskConfig
from feddqn.server_app import ServerConfig
from feddqn.client_app import DQNConfig
from feddqn import client_app, server_app

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../feddqn/conf")


class SimulationConfig(Config):
    num_supernodes: int


class FedDQNConfig(Config):
    client: DQNConfig
    server: ServerConfig
    task: TaskConfig
    simulation: SimulationConfig


@hydra.main(config_path="conf", config_name="cartpole", version_base=None)
def main(cfg_raw: DictConfig):
    cfg = FedDQNConfig.from_raw(cfg_raw)
    run_simulation(
        server_app=server_app.app(cfg.server),
        client_app=client_app.app(cfg.client, cfg.task),
        num_supernodes=cfg.simulation.num_supernodes,
        backend_name="ray",
    )


if __name__ == "__main__":
    main()
