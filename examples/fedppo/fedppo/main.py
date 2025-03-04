from flwr.simulation import run_simulation
import hydra

from fedppo import client_app, server_app


@hydra.main(config_path="conf", config_name="cartpole", version_base=None)
def main(cfg):
    run_simulation(
        server_app=server_app.app,
        client_app=client_app.app(cfg),
        num_supernodes=10,
        backend_name="ray",
    )
