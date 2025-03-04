from collections.abc import Sequence

import torch.nn as nn
from torch.distributions import OneHotCategorical
from torchrl.envs import EnvBase, GymEnv
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule, InteractionType


def make_env(env_name: str):
    return GymEnv(env_name)


def make_ppo_modules(
    env: EnvBase,
    num_cells: Sequence[int] | int = (120, 84),
    activation_class=nn.LeakyReLU,
):
    if not len(env.observation_spec.shape) <= 1:
        raise ValueError(
            f"Unsupported observation space with observation_spec {env.observation_spec}"
        )

    action_spec_space = env.action_spec.space

    if not isinstance(action_spec_space, CategoricalBox):
        raise ValueError(f"Unsupported action spec {env.action_spec}")

    actor_model = TensorDictModule(
        MLP(
            in_features=env.observation_spec["observation"].shape[0],
            out_features=action_spec_space.n,
            num_cells=num_cells,
            activation_class=activation_class,
        ),
        in_keys=tuple(env.observation_spec.keys()),
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=actor_model,
        in_keys=["logits"],
        out_keys=[env.action_key],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
        default_interaction_type=InteractionType.RANDOM,
    )

    value_model = MLP(
        in_features=env.observation_spec["observation"].shape[0],
        out_features=1,
        num_cells=num_cells,
        activation_class=activation_class,
    )

    value = ValueOperator(
        module=value_model, in_keys=tuple(env.observation_spec.keys())
    )

    return actor, value
