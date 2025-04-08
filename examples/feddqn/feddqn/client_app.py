import os
from pathlib import Path
from typing import Dict

import torch
from florl.client import EnvironmentClient
from florl.client.state import StatefulClient
from florl.common import Config, StateDict, transpose_dicts
from flwr.client import ClientApp
from flwr.common import Context
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import EnvBase
from torchrl.objectives import DQNLoss, SoftUpdate

from feddqn.task import TaskConfig, make_dqn_loss_modules, make_dqn_module, make_env

# FIXME: Currently we set a constant root directory, change if needed
ROOT_DIR = Path(os.path.join(os.path.dirname(__file__), "../feddqn_experiment"))


class DQNConfig(Config):
    # FIXME: don't actually do this
    node_id: int = 0
    minibatch_size: int = 128
    buffer_size: int = 10_000
    n_iterations: int = 100
    gamma: float = 0.99
    lr: float = 2.5e-4
    clip_grad_norm: float = 10
    target_update_interval: int = 10


class DQNClient(EnvironmentClient, StatefulClient):
    def __init__(
        self,
        env: EnvBase,
        node_id: int,
        actor_network: nn.Module,
        loss_module: DQNLoss,
        target_updater: SoftUpdate,
        minibatch_size: int = 128,
        buffer_size: int = 10_000,
        n_iterations: int = 10,
        gamma: float = 0.99,
        lr: float = 1e-3,
        clip_grad_norm: float = 1.0,
        target_update_interval: int = 10,
    ):
        print(node_id)
        print(ROOT_DIR)
        EnvironmentClient.__init__(self, env)
        StatefulClient.__init__(self, root_dir=ROOT_DIR, node_id=node_id)
        self.actor = actor_network
        self.node_id = node_id
        self.loss = loss_module
        self.target_updater = target_updater
        self.minibatch_size = minibatch_size
        self.n_iterations = n_iterations
        self.clip_grad_norm = clip_grad_norm
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optim = torch.optim.Adam(self.loss.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=self.working_dir,
                existsok=True,
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=minibatch_size,
        )

        self._env.to(self.device)
        self.actor.to(self.device)
        self.loss.to(self.device)
        # self.target_updater.to(self.device)
        self._parameter_container = nn.ModuleDict(
            {"actor": self.actor, "loss": self.loss}
        )

    @property
    def parameter_container(self) -> nn.Module:
        # return self.loss
        return self._parameter_container

    @property
    def _store_paths(self) -> Dict[str, Path]:
        return {
            # "replay_buffer": Path(self.working_dir / "replay_buffer.pt"),
            "loss_module": Path(self.working_dir / "loss_module.pt"),
        }

    def train_pre(self):
        # TODO: We may want to load the loss function from the disk
        pass

    def train_post(self):
        # TODO: We may want to load the loss function from the disk
        pass

    def train(self, parameters: StateDict, config):
        self.loss.load_state_dict(parameters)
        # collector = SyncDataCollector(
        #     create_env_fn=self._env,
        #     policy=self.actor,
        #     device=self.device,
        #     frames_per_batch=self.minibatch_size,
        #     total_frames=self.minibatch_size * self.n_iterations,
        # )

        # NOTE: SyncDataCollector already
        # handles stepping/generating action-reward pairs;
        # each step in the iterator takes a step.
        collector = SyncDataCollector(
            create_env_fn=self._env,
            policy=self.actor,
            frames_per_batch=self.minibatch_size,
            total_frames=self.n_iterations * self.minibatch_size,
            device=self.device,
            storing_device=self.device,
            max_frames_per_traj=-1,
            init_random_frames=10_000,
        )

        metrics: list[dict] = []
        target_update_counter = 0

        # NOTE: sampling_td is mnemonic for sampling tensor dict
        for sampling_td in collector:
            epoch_metrics = {}
            self.replay_buffer.extend(sampling_td.reshape(-1))

            # Sample replay buffer
            minibatch = self.replay_buffer.sample(batch_size=self.minibatch_size)

            # Compute loss
            self.optim.zero_grad()
            loss_vals = self.loss(minibatch)
            loss_vals["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.loss.parameters(), max_norm=self.clip_grad_norm
            )
            self.optim.step()

            # Update target network
            target_update_counter += 1
            if target_update_counter % self.target_update_interval == 0:
                self.target_updater.step()

            # Collect metrics
            with torch.no_grad():
                q_values = minibatch.get("action_value")
                epoch_metrics.update(
                    {
                        "train/loss/total": loss_vals["loss"].item(),
                        "train/q_values/mean": q_values.mean().item(),
                        "train/q_values/max": q_values.max().item(),
                        "train/q_values/min": q_values.min().item(),
                    }
                )

            metrics.append(epoch_metrics)

        return (
            len(self.replay_buffer),
            self.loss.state_dict(),
            transpose_dicts(metrics),
        )

    def evaluation(self, parameters, config):
        # self.loss.load_state_dict(parameters)
        # max_steps: int = config.get("max_steps", 500)
        # rollout = self._env.rollout(
        #     max_steps=max_steps, policy=self.actor, auto_cast_to_device=True
        # )
        # episode_reward = rollout.get(("next", "reward")).sum().item()
        # return max_steps, {"episode_reward": episode_reward}
        return 0


# Stub implementations for buffer serialization
def client_fn(context: Context, task_cfg: TaskConfig, client_cfg: DQNConfig):
    reference_env = make_env(task_cfg.name, mode="reference")
    train_env = make_env(task_cfg.name, mode="train")

    actor = make_dqn_module(reference_env)
    # IRL, this loss_module is only needed ONCE, in every subsequent rounds this is wasted computation.
    loss_module, target_updater = make_dqn_loss_modules(actor, client_cfg.gamma)

    print(client_cfg)

    client = DQNClient(
        env=train_env,
        actor_network=actor,
        loss_module=loss_module,
        target_updater=target_updater,
        **client_cfg.model_dump(),
    ).to_numpy()

    # TODO: Load existing replay buffer if available
    if False:  # Replace with actual condition
        client.replay_buffer.load_state_dict(
            torch.load(task_cfg["replay_buffer_store"])
        )

    # TODO: Load existing network weights if available
    if False:  # Replace with actual condition
        client.loss.load_state_dict(torch.load(task_cfg["network_store"]))

    return client


# Flower ClientApp
def app(client_cfg: DQNConfig, task_cfg: TaskConfig):
    def _client_fn(context: Context):
        return client_fn(context, task_cfg=task_cfg, client_cfg=client_cfg)

    return ClientApp(client_fn=_client_fn)
