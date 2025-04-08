from typing import TypedDict

import torch
from torch import nn
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives import ValueEstimators
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict.nn import TensorDictModule
from flwr.client import ClientApp
from flwr.common import Context
from florl.client import EnvironmentClient
from florl.common import Config, transpose_dicts

from fedppo.task import TaskConfig, make_env, make_ppo_modules


class PPOConfig(Config):
    minibatch_size: int = 128
    n_minibatches: int = 4
    n_update_epochs: int = 4
    n_iterations: int = 10
    gae_gamma: float = 0.99
    gae_lmbda: float = 0.95
    clip_grad_norm: float = 1.0
    lr: float = 2.5e-4
    normalize_advantage: bool = True
    clip_epsilon: float = 0.2


class PPOClient(EnvironmentClient):
    def __init__(
        self,
        env: EnvBase,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        minibatch_size: int = 128,
        n_minibatches: int = 4,
        n_update_epochs: int = 4,
        n_iterations: int = 10,
        gae_gamma: float = 0.99,
        gae_lmbda: float = 0.95,
        clip_grad_norm: float = 1.0,
        lr: float = 2.5e-4,
        normalize_advantage: bool = True,
        clip_epsilon: float = 0.2,
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
            normalize_advantage=normalize_advantage,
            clip_epsilon=clip_epsilon,
            loss_critic_type="l2",
        )
        self.loss.make_value_estimator(
            ValueEstimators.GAE, gamma=gae_gamma, lmbda=gae_lmbda
        )

        self.optim = torch.optim.Adam(self.loss.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.batch_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.minibatch_size,
        )

        self._env.to(self.device)
        self.actor.to(self.device)
        self.loss.to(self.device)

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

        metrics: list[dict] = []

        for sampling_td in collector:
            epoch_metrics = {}
            with torch.no_grad():
                self.loss.value_estimator(
                    sampling_td,
                    params=self.loss.critic_network_params,
                    target_params=self.loss.target_critic_network_params,
                )

            dones = sampling_td.get(("next", "done"))
            rewards = sampling_td.get(("next", self._env.reward_key))
            episode_rewards = (
                sampling_td.get(("next", "episode_reward"))[dones]
                if dones.any()
                else torch.zeros(())
            )
            epoch_metrics.update(
                {
                    "train/reward/reward_mean": rewards.mean().item(),
                    "train/reward/reward_min": rewards.min().item(),
                    "train/reward/reward_max": rewards.max().item(),
                    "train/reward/episode_reward_mean": episode_rewards.mean().item(),
                    "train/reward/episode_reward_min": episode_rewards.min().item(),
                    "train/reward/episode_reward_max": episode_rewards.max().item(),
                }
            )

            self.replay_buffer.extend(sampling_td.reshape(-1))

            loss_metrics: dict[str, float] = {}
            for _ in range(self.n_update_epochs):
                for _ in range(self.n_minibatches):
                    self.optim.zero_grad()

                    minibatch: TensorDict = self.replay_buffer.sample()

                    loss_vals = self.loss(minibatch)
                    total_loss = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    for k, v in loss_vals.items():
                        if v is not None:
                            loss_metrics[f"train/loss/{k}"] = (
                                loss_metrics.get(k, 0) + v.item() / self.n_update_epochs
                            )

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.loss.parameters(), max_norm=self.clip_grad_norm
                    )

                    self.optim.step()

            collector.update_policy_weights_()

            epoch_metrics.update(loss_metrics)
            metrics.append(epoch_metrics)

        return (self.total_frames, self.loss.state_dict(), transpose_dicts(metrics))

    def evaluation(self, parameters, config):
        self.loss.load_state_dict(parameters)

        max_steps: int = config.get("max_steps", 500)
        rollout: TensorDict = self._env.rollout(
            max_steps=max_steps, policy=self.actor, auto_cast_to_device=True
        )
        episode_reward = rollout.get(("next", "reward")).sum().item()
        return max_steps, {"episode_reward": episode_reward}


def client_fn(context: Context, task_cfg: TaskConfig, client_cfg: PPOConfig):
    reference_env = make_env(task_cfg.name, mode="reference")
    train_env = make_env(task_cfg.name, mode="train")

    actor, value = make_ppo_modules(reference_env)

    return PPOClient(
        env=train_env,
        actor_network=actor,
        value_network=value,
        **client_cfg.model_dump(),
    ).to_numpy()


# Flower ClientApp


def app(client_cfg: PPOConfig, task_cfg: TaskConfig):
    def _client_fn(context: Context):
        return client_fn(context, task_cfg=task_cfg, client_cfg=client_cfg)

    return ClientApp(client_fn=_client_fn)
