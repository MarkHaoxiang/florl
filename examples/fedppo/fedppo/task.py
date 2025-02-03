from collections.abc import Sequence

import torch.nn as nn
from torchrl.envs import EnvBase, GymEnv
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.modules import MLP, QValueActor


def make_env():
    return GymEnv("CartPole-v1")


def make_dqn_modules(
    env: EnvBase,
    num_cells: Sequence[int] | int = (120, 84),
    activation_class=nn.LeakyReLU,
):
    if not len(env.observation_spec.shape == 1):
        raise ValueError(
            f"Unsupported observation space with observation_spec {env.observation_spec}"
        )

    action_spec_space = env.action_spec.space
    if not isinstance(action_spec_space, CategoricalBox):
        raise ValueError(f"Unsupported action spec {env.action_spec}")

    qvalue_actor_model = MLP(
        in_features=env.observation_spec["observation"].shape[0],
        out_features=action_spec_space.n,
        num_cells=num_cells,
        activation_class=activation_class,
    )

    qvalue_actor = QValueActor(
        module=qvalue_actor_model,
        in_keys=env.observation_spec.keys(),
        spec=env.action_spec,
    )

    return qvalue_actor
