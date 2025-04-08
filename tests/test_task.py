import warnings
import torch
import torch.nn as nn
from torchrl.data.replay_buffers import ReplayBuffer, ListStorage

from florl.client import EnvironmentClient
from florl.task import OnlineAlgorithm
from florl.task.offline import RandomPartitionSamplerWithoutReplacement


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
        import gymnasium  # noqa: F401
    except ImportError:
        warnings.warn("gymnasium is not available. Skipping test.")
        return

    from florl.task.libs.gymnasium import (
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


def test_partition_sampler():
    partition = torch.arange(0, 10, 2)
    rb = ReplayBuffer(
        storage=ListStorage(max_size=1000),
        sampler=RandomPartitionSamplerWithoutReplacement(partition=partition),
        batch_size=5,
    )
    rb.extend(torch.arange(100).reshape(-1, 1))

    # Test sampling from the partition (should be a subset of the partition)
    samples = rb.sample()
    assert all(idx in partition for idx in samples), (
        "Sampled indices are not in the partition"
    )

    # Test empty partition (should raise an error)
    try:
        rb.sampler._partition = torch.tensor([])
        rb.sample()
        assert False, "Empty partition should raise an error"
    except ValueError:
        pass
