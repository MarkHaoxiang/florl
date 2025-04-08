from collections.abc import Sequence
from typing import Literal

from torch import nn
from torchrl.envs import EnvBase, RewardSum
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    TransformedEnv,
)
from torchrl.modules import MLP, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate
from florl.common import Config


class TaskConfig(Config):
    name: str
    num_cells: Sequence[int] | int


def make_env(
    name: str,
    mode: Literal["reference", "train"] = "train",
):
    env = GymEnv(name)
    if mode == "train":
        env = TransformedEnv(
            env,
            RewardSum(in_keys=env.reward_keys, out_keys=["episode_reward"]),  # noqa: F821
        )
    return env


def make_dqn_module(
    env: EnvBase, num_cells: Sequence[int] | int = (120, 84), activation_class=nn.ELU
):
    action_spec_space = env.action_spec.space
    assert action_spec_space is not None

    value_model = MLP(
        in_features=env.observation_spec["observation"].shape[0],
        out_features=1,
        num_cells=num_cells,
        activation_class=activation_class,
    )

    actor = QValueActor(
        value_model, in_keys=tuple(env.observation_spec.keys()), spec=env.action_spec
    )

    return actor


def make_dqn_loss_modules(actor, gamma):
    loss_module = DQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater
