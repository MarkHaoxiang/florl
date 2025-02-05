import torch
from torch import nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.ppo import ClipPPOLoss, GAE
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict.nn import TensorDictModule
from flwr.client import ClientApp
from flwr.common import Context
from florl.client import EnvironmentClient

from fedppo.task import make_env, make_dqn_modules


# Define Flower Client and client_fn
class PPOClient(EnvironmentClient):
    def __init__(
        self,
        env: EnvBase,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        minibatch_size: int = 1024,
        n_minibatches: int = 10,
        n_update_epochs: int = 4,
        n_iterations: int = 1,
        gae_gamma: float = 0.99,
        gae_lmbda: float = 0.95,
        clip_grad_norm: float = 1.0,
        lr: float = 3e-4,
    ):
        super().__init__(env)
        self.actor = actor_network
        self.critic = value_network
        self.minibatch_size = minibatch_size
        self.n_minibatches = n_minibatches
        self.n_update_epochs = n_update_epochs
        self.n_iterations = n_iterations
        self.batch_size = self.minibatch_size * n_minibatches
        self.total_frames = self.batch_size * self.n_iterations
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.loss = ClipPPOLoss(
            actor_network,
            value_network,
        )

        self.optim = torch.optim.Adam(self.loss.parameters(), lr=lr)

        self.advantage = GAE(
            gamma=gae_gamma,
            lmbda=gae_lmbda,
            value_network=value_network,
            advantage_key=True,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.batch_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.minibatch_size,
        )

    @property
    def parameter_container(self) -> nn.Module:
        return self.loss

    def train(self, parameters, config):
        self.loss.load_state_dict(parameters)

        collector = SyncDataCollector(
            create_env_fn=self._env,
            policy=self.actor,
            device=self.device,
            frames_per_batch=self.batch_size,
            total_frames=self.total_frames,
        )

        for sampling_td in collector:
            self.advantage(sampling_td)
            self.replay_buffer.extend(sampling_td.reshape(-1))

            for _ in range(self.n_update_epochs):
                for _ in range(self.n_minibatches):
                    self.optim.zero_grad()

                    minibatch: TensorDict = self.replay_buffer.sample()
                    loss_vals = self.loss()
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.loss.parameters(), max_norm=self.clip_grad_norm
                    )

                    self.optim.step()

        return (self.total_frames, self.loss.state_dict(), {})  # TODO mark: Metrics


def client_fn(context: Context):
    # Load model and data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    env = make_env()
    qvalue_actor = make_dqn_modules(env)

    # Return Client instance
    raise PPOClient(env=env, qvalue_actor=qvalue_actor)


# Flower ClientApp
app = ClientApp(client_fn)
