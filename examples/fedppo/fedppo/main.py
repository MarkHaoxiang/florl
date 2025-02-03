from flwr.simulation import run_simulation

from fedppo import client_app, server_app


def main():
    run_simulation(
        server_app=server_app.app,
        client_app=client_app,
        num_supernodes=10,
        backend_name="ray",
    )


if __name__ == "__main__":
    main()
