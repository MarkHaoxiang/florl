import os

from flwr.simulation import run_simulation
from florl.common import Config
from omegaconf import DictConfig
import hydra

from fedppo.task import TaskConfig
from fedppo.server_app import ServerConfig
from fedppo.client_app import PPOConfig
from fedppo import client_app, server_app


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../fedppo/conf")


class SimulationConfig(Config):
    num_supernodes: int


class FedPPOConfig(Config):
    client: PPOConfig
    server: ServerConfig
    task: TaskConfig
    simulation: SimulationConfig


@hydra.main(config_path="conf", config_name="cartpole", version_base=None)
def main(cfg_raw: DictConfig):
    cfg = FedPPOConfig.from_raw(cfg_raw)
    run_simulation(
        server_app=server_app.app(cfg.server),
        client_app=client_app.app(cfg.client, cfg.task),
        num_supernodes=cfg.simulation.num_supernodes,
        backend_name="ray",
    )


if __name__ == "__main__":
    main()
