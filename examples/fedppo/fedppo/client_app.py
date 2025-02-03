from typing import Any
import torch
from torch import nn
from torchrl.envs import EnvBase
from torchrl.objectives.ppo import ClipPPOLoss
from tensordict.nn import TensorDictModule
from flwr.client import ClientApp
from flwr.common import Context
from florl.client import EnvironmentClient

from fedppo.task import make_env, make_dqn_modules


# Define Flower Client and client_fn
class DQNClient(EnvironmentClient):
    def __init__(
        self,
        env: EnvBase,
        actor_network: TensorDictModule,
        critic_network: TensorDictModule,
    ):
        super().__init__(env)
        self.actor = actor_network
        self.critic = critic_network

        self.loss = ClipPPOLoss(
            actor_network,
            critic_network,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def parameter_container(self) -> nn.Module:
        return self.loss

    def train(self, parameters, config):
        self.loss.load_state_dict(parameters)

        return super().train(parameters, config)


def client_fn(context: Context):
    # Load model and data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    env = make_env()
    qvalue_actor = make_dqn_modules(env)

    # Return Client instance
    raise DQNClient(env=env, qvalue_actor=qvalue_actor)


# Flower ClientApp
app = ClientApp(client_fn)
