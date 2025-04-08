import warnings
import torch.nn as nn
from florl.client import EnvironmentClient
from florl.task import OnlineAlgorithm


class EmptyModule(nn.Module):
    def forward(self, x):
        return x


class PlaceholderClient(EnvironmentClient):
    def __init__(self, env):
        super().__init__(env)

    @property
    def parameter_container(self):
        return EmptyModule()

    def train(self, parameters, config):
        return 0, {}, {}

    def evaluate(self, parameters, config):
        return 0, {}


def test_classical_control():
    # Skip if gymnasium is not available
    try:
        import gymnasium
    except ImportError:
        warnings.warn("gymnasium is not available. Skipping test.")
        return

    from florl.task.gymnasium import (
        ClassicalControlDiscrete,
        ClassicalControlContinuous,
    )

    class PlaceholderAlgorithm(OnlineAlgorithm):
        def make_client(self, train_env, eval_env, reference_env):
            # Placeholder implementation

            return PlaceholderClient(train_env)

        def make_server(self):
            raise NotImplementedError()

    # Test compilation of tasks with a placeholder algorithm
    for task in ClassicalControlDiscrete.tasks + ClassicalControlContinuous.tasks:
        algorithm = PlaceholderAlgorithm()
        _ = task.compile(algorithm)
